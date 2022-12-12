// lab2.h
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include "common.h"
#include <cstdio>

#define NMAX 6000000
#define START_TIMES 12
#define EXTENSION ".log"
#define CHUNK 100

using namespace std;

void test_time(
    void (*method)(TYPE_PARAM *c, TYPE_PARAM *a, TYPE_PARAM *b, int size, bool is_q),
    std::ostream &out,
    bool is_q = false,
    bool is_print = false
) {
    for (int t = 0; t < START_TIMES; ++t) {
        TYPE_PARAM *a_vector = init_array(create_array(NMAX), NMAX);
        TYPE_PARAM *b_vector = init_array(create_array(NMAX), NMAX);
        TYPE_PARAM *c_vector = create_array(NMAX);
        double start = omp_get_wtime();
        method(c_vector, a_vector, b_vector, NMAX, is_q);
        double result_time = (omp_get_wtime() - start) * 1000; // time in ms
        out << result_time << std::endl;
        delete_array(a_vector);
        delete_array(b_vector);
        delete_array(c_vector);
        if (is_print) {
            printf("c = ");
            print_array(c_vector, NMAX);
            printf("TIME OF WORK IS %f ms\n", result_time);
        }
    }
}

void static_sum(TYPE_PARAM *c, TYPE_PARAM *a, TYPE_PARAM *b, int size, bool is_q = false) {
    int i;
    if (is_q) {
        #pragma omp parallel for schedule(static, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
    } else {
        #pragma omp parallel for schedule(static, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = sum_with_q(a[i], b[i]);
        }
    }
}


void dynamic_sum(TYPE_PARAM *c, TYPE_PARAM *a, TYPE_PARAM *b, int size, bool is_q = false) {
    int i;
    if (is_q) {
        #pragma omp parallel for schedule(dynamic, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
    } else {
        #pragma omp parallel for schedule(dynamic, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = sum_with_q(a[i], b[i]);
        }
    }
}

void guided_sum(TYPE_PARAM *c, TYPE_PARAM *a, TYPE_PARAM *b, int size, bool is_q = false) {
    int i;
    if (is_q) {
        #pragma omp parallel for schedule(guided, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
    } else {
        #pragma omp parallel for schedule(guided, CHUNK) shared(a, b, c) private(i)
        for (i = 0; i < size; ++i) {
            c[i] = sum_with_q(a[i], b[i]);
        }
    }
}
