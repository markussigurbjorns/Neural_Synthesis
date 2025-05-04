// conv_vae.c 
// ----------------------------------------------------------------------------
//   * mono audio 1 × 2048 samples
//   * Encoder: Conv‑ReLU (kernel 4, stride 4) → flatten → Dense → μ | logσ²
//   * Decoder: Dense → reshape (16×512) → Transpose‑Conv1D (k 4, s 4)
//   * Loss   : mean‑BCE + β‑KL   (β warm‑up over first 500 epochs)
//   * Optim  : SGD with grad‑clip ±1   (lr 3 × 10⁻⁴)
// ----------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <raylib.h>

/* ─── configuration ─────────────────────────────────────────────── */
#define LEN 2048 /* audio frame length               */
#define CH_IN 1 /* mono                             */
#define LATENT_DIM 2 /* z dimensionality                 */
#define EPS 1e-7

/* ─── utilities ─────────────────────────────────────────────────── */
static inline double randn(void) {
    static double n2;
    static int h = 0;
    if (h) {
        h = 0;
        return n2;
    }
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1;
        v = 2.0 * rand() / RAND_MAX - 1;
        s = u * u + v * v;
    } while (s == 0 || s >= 1);
    double m = sqrt(-2 * log(s) / s);
    n2 = v * m;
    h = 1;
    return u * m;
}
static inline double relu(double x) {
    return x > 0 ? x : 0;
}

static inline double d_relu(double x) {
    return x > 0 ? 1 : 0;
}

static inline double sigmoid(double x) {
    if (x > 80) return 1;
    if (x < -80) return 0;
    return 1 / (1 + exp(-x));
}

/* ─── Conv1D / Transposed‑Conv1D struct ─────────────────────────── */
typedef struct {
    int in_ch, out_ch, kernel, stride, in_len, out_len;
    double * w, * b, * mask; /* mask used only for encoder ReLU */
    int transpose; /* 0 = conv, 1 = t‑conv            */
} Conv1D;

static Conv1D * conv_create(int in_ch, int out_ch, int in_len, int kernel, int stride, int transpose) {
    Conv1D * c = malloc(sizeof * c);
    c -> in_ch = in_ch;
    c -> out_ch = out_ch;
    c -> kernel = kernel;
    c -> stride = stride;
    c -> in_len = in_len;
    c -> transpose = transpose;
    c -> out_len = transpose ? (in_len - 1) * stride + kernel :
        (in_len - kernel) / stride + 1;
    int ws = in_ch * out_ch * kernel;
    c -> w = malloc(ws * sizeof(double));
    c -> b = calloc(out_ch, sizeof(double));
    c -> mask = transpose ? NULL : calloc(out_ch * c -> out_len, sizeof(double));
    double std = sqrt(2.0 / (in_ch * kernel));
    for (int i = 0; i < ws; ++i) c -> w[i] = randn() * std;
    return c;
}

/* forward pass */
static void conv_forward(const Conv1D * c,
    const double * x, double * y) {
    if (!c -> transpose) {
        /* standard convolution */
        for (int oc = 0; oc < c -> out_ch; ++oc)
            for (int t = 0; t < c -> out_len; ++t) {
                double z = c -> b[oc];
                int xs = t * c -> stride;
                for (int ic = 0; ic < c -> in_ch; ++ic)
                    for (int k = 0; k < c -> kernel; ++k)
                        z += c -> w[((oc * c -> in_ch + ic) * c -> kernel) + k] * x[ic * c -> in_len + xs + k];
                double a = relu(z);
                y[oc * c -> out_len + t] = a;
                c -> mask[oc * c -> out_len + t] = a > 0;
            }
    } else {
        /* transpose conv (fractionally‑strided) */
        memset(y, 0, c -> out_ch * c -> out_len * sizeof(double));
        for (int ic = 0; ic < c -> in_ch; ++ic)
            for (int t = 0; t < c -> in_len; ++t) {
                double v = x[ic * c -> in_len + t]; /* logits, no activation */
                int ys = t * c -> stride;
                    for (int oc = 0; oc < c -> out_ch; ++oc) {
                        double * w = & c -> w[((ic * c -> out_ch + oc) * c -> kernel)];
                        double * row = & y[oc * c -> out_len + ys];
                        for (int k = 0; k < c -> kernel; ++k) row[k] += w[k] * v;
                    }
            }
        for (int oc = 0; oc < c -> out_ch; ++oc)
            for (int t = 0; t < c -> out_len; ++t)
                y[oc * c -> out_len + t] += c -> b[oc];

    }
}

/* gradient clip helper */
static void clip(double * g, int n) {
  for (int i = 0; i < n; ++i) {
    if (g[i] > 1) g[i] = 1;
    else if (g[i] < -1) g[i] = -1;
  }
}

/* backward pass */
static void conv_backward(Conv1D * c,
  const double * x,
    const double * delta, double lr, double * delta_prev) {
  int ws = c -> in_ch * c -> out_ch * c -> kernel;
  memset(delta_prev, 0, c -> in_ch * c -> in_len * sizeof(double));
  double * dW = calloc(ws, sizeof(double)), * dB = calloc(c -> out_ch, sizeof(double));

  if (!c -> transpose) {
    /* encoder conv */
    for (int oc = 0; oc < c -> out_ch; ++oc)
      for (int t = 0; t < c -> out_len; ++t) {
        double d = delta[oc * c -> out_len + t] * c -> mask[oc * c -> out_len + t];
        int xs = t * c -> stride;
        dB[oc] += d;
        for (int ic = 0; ic < c -> in_ch; ++ic)
          for (int k = 0; k < c -> kernel; ++k) {
            int widx = ((oc * c -> in_ch + ic) * c -> kernel) + k;
            int xi = ic * c -> in_len + xs + k;
            dW[widx] += d * x[xi];
            delta_prev[xi] += c -> w[widx] * d;
          }
      }
  } else {
    /* decoder transpose conv */
    for (int ic = 0; ic < c -> in_ch; ++ic)
      for (int t = 0; t < c -> in_len; ++t) {
        double v = x[ic * c -> in_len + t];
        int ys = t * c -> stride;
        for (int oc = 0; oc < c -> out_ch; ++oc) {
          double * w = & c -> w[((ic * c -> out_ch + oc) * c -> kernel)];
          for (int k = 0; k < c -> kernel; ++k) {
            int yidx = oc * c -> out_len + ys + k;
            if (yidx >= c -> out_ch * c -> out_len) continue;
            double d = delta[yidx];
            dW[((ic * c -> out_ch + oc) * c -> kernel) + k] += d * v;
            delta_prev[ic * c -> in_len + t] += w[k] * d;
          }
        }
      }
    for (int oc = 0; oc < c -> out_ch; ++oc)
      for (int t = 0; t < c -> out_len; ++t)
        dB[oc] += delta[oc * c -> out_len + t];
  }
  clip(dW, ws);
  clip(dB, c -> out_ch); /* keep bias under control */
  for (int i = 0; i < ws; ++i) c -> w[i] -= lr * dW[i];
  for (int oc = 0; oc < c -> out_ch; ++oc) c -> b[oc] -= lr * dB[oc];
  free(dW);
  free(dB);
}

static void conv_free(Conv1D * c) {
  free(c -> w);
  free(c -> b);
  if (c -> mask) free(c -> mask);
  free(c);
}

/* ─── Dense layer helpers ───────────────────────────────────────── */
typedef struct {
  int in , out;
  double * w, * b;
}
Dense;
static Dense * dense_create(int in , int out) {
  Dense * d = malloc(sizeof * d);
  d -> in = in;
  d -> out = out;
  d -> w = malloc(in * out * sizeof(double));
  d -> b = calloc(out, sizeof(double));
  double std = sqrt(2.0 / in);
  for (int i = 0; i < in * out; ++i) d -> w[i] = randn() * std;
  return d;
}
static void dense_forward(const Dense * d,
 const double * x, double * y) {
  for (int o = 0; o < d -> out; ++o) {
    double z = d -> b[o];
    for (int i = 0; i < d -> in; ++i) z += d -> w[o * d -> in + i] * x[i];
    y[o] = z;
  }
}
static void dense_backward(Dense * d,
  const double * x,
    const double * delta, double lr, double * delta_prev) {
  if (delta_prev) {
    memset(delta_prev, 0, d -> in * sizeof(double));
    for (int i = 0; i < d -> in; ++i) {
      double s = 0;
      for (int o = 0; o < d -> out; ++o) s += d -> w[o * d -> in + i] * delta[o];
      delta_prev[i] = s;
    }
  }
  for (int o = 0; o < d -> out; ++o) {
    for (int i = 0; i < d -> in; ++i) d -> w[o * d -> in + i] -= lr * delta[o] * x[i];
    d -> b[o] -= lr * delta[o];
  }
}
static void dense_free(Dense * d) {
  free(d -> w);
  free(d -> b);
  free(d);
}

/* ─── training ---------------------------------------------------- */
int main(void) {
    srand((unsigned) time(NULL));
   
    /* encoder */
    Conv1D*enc_conv1 = conv_create(CH_IN, 32, LEN, 5, 2, 0); /* 2048 -> 1024 */
    Conv1D*enc_conv2 = conv_create(32, 64, enc_conv1->out_len, 5, 2, 0); /* 1024 -> 512 */
    int ENC_LEN = enc_conv2->out_len;
    int FLAT = 64*ENC_LEN;
    Dense*enc_fc = dense_create(FLAT, LATENT_DIM * 2);
   
    /* decoder */
    Dense*dec_fc = dense_create(LATENT_DIM, 64 * ENC_LEN);
    Conv1D*dec_tconv1 = conv_create(64, 32, ENC_LEN, 5, 2, 1); /* 512 -> 1024 */
    Conv1D*dec_tconv2 = conv_create(32, 1, dec_tconv1->out_len, 5, 2, 1); /* 1024 -> 2048 */
      
    /* toy sine data */
    static double data[4][LEN];

    for (int i = 0; i<4; i++) {
        float phase = 0;
        const float phaseIncrement = (2.0f * (float)M_PI * 440) / 44100;
        for(int j = 0; j < LEN; j++) {
            data[i][j] = sinf(phase);
            phase += phaseIncrement;
            if(phase >= 2.0f * (float)M_PI) {
                phase -= 2.0f * (float)M_PI;
            }
        }
    }
   
    const double lr = 3e-4;
    const int epochs = 10000;
   
    /* buffers */
    double * enc_act1 = malloc(32 * enc_conv1->out_len * sizeof(double));
    double * enc_act2 = malloc(64 * enc_conv2->out_len * sizeof(double));
    double lat[LATENT_DIM * 2], mu[LATENT_DIM], lv[LATENT_DIM];
    double z[LATENT_DIM], eps[LATENT_DIM];
    double * dec_h1 = malloc(64 * enc_conv2->out_len * sizeof(double));
    double * dec_h2 = malloc(32 * dec_tconv1->out_len * sizeof(double));
    double * logits = malloc(LEN * sizeof(double));
    double * d_logits = malloc(LEN * sizeof(double));
    double * delta_d1 = malloc(32 * dec_tconv1->out_len * sizeof(double));
    double * delta_d2 = malloc(64 * enc_conv2->out_len * sizeof(double));
    double * delta_enc_act1 = malloc(32 * enc_conv1->out_len * sizeof(double));
    double * delta_enc_act2 = malloc(64 * enc_conv2->out_len * sizeof(double));
    double * delta_prev_conv = malloc(CH_IN * LEN * sizeof(double));
    double delta_z[LATENT_DIM], delta_lat[LATENT_DIM * 2];
   
    for (int e = 0; e <= epochs; ++e) {
      double loss_sum = 0;
      for (int n = 0; n < 4; ++n) {
        /* ---------- forward ---------- */
        conv_forward(enc_conv1, data[n], enc_act1);
        conv_forward(enc_conv2, enc_act1, enc_act2);
        dense_forward(enc_fc, enc_act2, lat);
        memcpy(mu, lat, LATENT_DIM * sizeof(double));
        memcpy(lv, lat + LATENT_DIM, LATENT_DIM * sizeof(double));
        for (int i = 0; i < LATENT_DIM; ++i) {
          eps[i] = randn();
          double std = exp(0.5 * lv[i]);
          z[i] = mu[i] + std * eps[i];
        }
        dense_forward(dec_fc, z, dec_h1);
        conv_forward(dec_tconv1, dec_h1, dec_h2);
        conv_forward(dec_tconv2, dec_h2, logits);
   
        /* ---------- loss ------------ */
        double rec = 0;
        for (int i = 0; i < LEN; ++i) {
          double e = logits[i] - data[n][i];
          rec += 0.5 * e * e;             /* MSE */
        }
        rec /= LEN;
        double kl = 0;
        for (int i = 0; i < LATENT_DIM; ++i) kl += 0.5 * (mu[i] * mu[i] + exp(lv[i]) - lv[i] - 1);
        double beta = e < 500 ? e / 500.0 : 1.0;
        double L = rec + beta * kl;
        loss_sum += L;
   
        /* ---------- backward ---------- */
        for (int i = 0; i < LEN; ++i) d_logits[i] = (logits[i] - data[n][i]) / LEN; /* mean BCE derivative */
   
        /* decoder t‑conv → dec_h */
        conv_backward(dec_tconv2, dec_h2, d_logits, lr, delta_d1);
        conv_backward(dec_tconv1, dec_h1, delta_d1, lr, delta_d2);
   
        /* dense decoder */
        memset(delta_z, 0, sizeof delta_z);
        dense_backward(dec_fc, z, delta_d2, lr, delta_z);
   
        /* add KL gradients */
        for (int i = 0; i < LATENT_DIM; ++i) {
          delta_lat[i] = delta_z[i] + beta * mu[i];
          delta_lat[i + LATENT_DIM] = delta_z[i] * eps[i] * 0.5 * exp(0.5 * lv[i]) + 0.5 * beta * (exp(lv[i]) - 1);
        }
   
        /* encoder dense */
        memset(delta_enc_act1, 0, 32 * enc_conv1->out_len * sizeof(double));
        memset(delta_enc_act2, 0, 64 * enc_conv2->out_len * sizeof(double));

        dense_backward(enc_fc, enc_act2, delta_lat, lr, delta_enc_act2);
   
        /* encoder conv */
        conv_backward(enc_conv2, enc_act1, delta_enc_act2, lr, delta_enc_act1);
        conv_backward(enc_conv1, data[n], delta_enc_act1, lr, delta_prev_conv);
      }
      if (e % 100 == 0) printf("epoch %d  loss %.4f\n", e, loss_sum / 4);
    }
   
    /* ---- generate samples ---- */
    puts("Generated samples:");
    for (int s = 0; s < 4; ++s) {
      for (int i = 0; i < LATENT_DIM; ++i) z[i] = randn();
      dense_forward(dec_fc, z, dec_h1);
      conv_forward(dec_tconv1, dec_h1, dec_h2);
      conv_forward(dec_tconv2, dec_h2, logits);
      printf("%d: [%.3f, %.3f, %.3f, %.3f ...]\n", s + 1,
        logits[0], logits[1], logits[2], logits[3]);
    }

    /* ----------  Visualize Generated Samples -------------*/
   
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    Vector2* points = (Vector2*)malloc(LEN * sizeof(Vector2));  // Allocate full LEN

    int x = 600;
    int y = 600;
    InitWindow(x, y, "julius - Audio Analyzing and Visualizing");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        int currentScreenWidth = GetScreenWidth();
        int currentScreenHeight = GetScreenHeight();

        for (int i = 0; i < LATENT_DIM; ++i) z[i] = randn();
        dense_forward(dec_fc, z, dec_h1);
        conv_forward(dec_tconv1, dec_h1, dec_h2);
        conv_forward(dec_tconv2, dec_h2, logits);
        
        float min = logits[0];
        float max = logits[0];
        for (int i = 1; i < LEN; i++) {
            if (logits[i] < min) min = logits[i];
            if (logits[i] > max) max = logits[i];
        }
        float range = max - min;
        
        for (int i = 0; i < LEN; i++) {
            points[i].x = (float)i * currentScreenWidth / (float)LEN;
            float norm = (logits[i] - min) / range;  // normalize to [0, 1]
            points[i].y = (1.0f - norm) * currentScreenHeight; // flip to match screen coordinates
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawFPS(0, 0);
        DrawLineStrip(points, LEN, RED);
        EndDrawing();
    }

    free(points);
    CloseWindow();

    /* ---- cleanup ---- */
    conv_free(enc_conv1);
    conv_free(enc_conv2);
    conv_free(dec_tconv1);
    conv_free(dec_tconv2);
    dense_free(enc_fc);
    dense_free(dec_fc);
    free(enc_act1);
    free(enc_act2);
    free(dec_h1);
    free(dec_h2);
    free(logits);
    free(d_logits);
    free(delta_d1);
    free(delta_d2);
    free(delta_enc_act1);
    free(delta_enc_act2);
    free(delta_prev_conv);
    return 0;
}
