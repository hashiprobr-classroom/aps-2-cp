#include <math.h>

#include "fourier.h"

void normalize(complex s[], int n) {
    for (int k = 0; k < n; k++) {
        s[k].a /= n;
        s[k].b /= n;
    }
}

void nft(complex s[], complex t[], int n, int sign) {
    for (int k = 0; k < n; k++) {
        t[k].a = 0;
        t[k].b = 0;

        for (int j = 0; j < n; j++) {
            double x = sign * 2 * PI * k * j / n;

            double cosx = cos(x);
            double sinx = sin(x);

            t[k].a += s[j].a * cosx - s[j].b * sinx;
            t[k].b += s[j].a * sinx + s[j].b * cosx;
        }
    }
}

void nft_forward(complex s[], complex t[], int n) {
    nft(s, t, n, -1);
}

void nft_inverse(complex t[], complex s[], int n) {
    nft(t, s, n, 1);
    normalize(s, n);
}

void fft(complex s[], complex t[], int n, int sign) {
    if (n == 1) {
        t[0].a = s[0].a;
        t[0].b = s[0].b;
        return;
    }

    int half = n / 2;
    complex sp[half], si[half];
    complex tp[half], ti[half];

    for (int k = 0; k < half; k++) {
        sp[k] = s[2 * k];
        si[k] = s[2 * k + 1];
    }

    fft(sp, tp, half, sign);
    fft(si, ti, half, sign);

    for (int k = 0; k < half; k++) {
        double x = sign * 2 * PI * k / n;
        double cosx = cos(x);
        double sinx = sin(x);
        double ra = ti[k].a * cosx - ti[k].b * sinx;
        double rb = ti[k].a * sinx + ti[k].b * cosx;

        t[k].a = tp[k].a + ra;
        t[k].b = tp[k].b + rb;
        t[k + half].a = tp[k].a - ra;
        t[k + half].b = tp[k].b - rb;
    }
}

void fft_forward(complex s[], complex t[], int n) {
    fft(s, t, n, -1);
}

void fft_inverse(complex t[], complex s[], int n) {
    fft(t, s, n, 1);
    normalize(s, n);
}

void fft_forward_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex temp[MAX_SIZE], temp2[MAX_SIZE];

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            temp[y] = matrix[y][x];
        }
        fft_forward(temp, temp2, height);
        for (int y = 0; y < height; y++) {
            matrix[y][x] = temp2[y];
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            temp[x] = matrix[y][x];
        }
        fft_forward(temp, temp2, width);
        for (int x = 0; x < width; x++) {
            matrix[y][x] = temp2[x];
        }
    }
}

void fft_inverse_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex temp[MAX_SIZE], temp2[MAX_SIZE];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            temp[x] = matrix[y][x];
        }
        fft_inverse(temp, temp2, width);
        for (int x = 0; x < width; x++) {
            matrix[y][x] = temp2[x];
        }
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            temp[y] = matrix[y][x];
        }
        fft_inverse(temp, temp2, height);
        for (int y = 0; y < height; y++) {
            matrix[y][x] = temp2[y];
        }
    }
}

void filter(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height, int flip) {
    int center_x = width / 2;
    int center_y = height / 2;

    double variance = -2 * SIGMA * SIGMA;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = center_x - (x + center_x) % width;
            int dy = center_y - (y + center_y) % height;

            double d = dx * dx + dy * dy;
            double g = exp(d / variance);

            if (flip) {
                g = 1 - g;
            }

            output[y][x].a = g * input[y][x].a;
            output[y][x].b = g * input[y][x].b;
        }
    }
}

void filter_lp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 0);
}

void filter_hp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 1);
}
