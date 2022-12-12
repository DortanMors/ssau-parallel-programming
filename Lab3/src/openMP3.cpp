#include "lab3omp.h"

using namespace std;

static bool is_print = false;

void log(const char* text, bool param);

// argv[1] - количество потоков
// argv[2] - флаг q
// argv[3] - путь до лога
// argv[4] - флаг печати
int main(int argc, char* argv[])
{
	int threads_count = argc > 1 ? stoi(string(argv[1])) : 1;
    omp_set_num_threads(threads_count);
    bool is_q = argc > 2 && string(argv[2]) == "q";
    string path(argc > 3 ? string(argv[3]) : "results");
    is_print = argc > 4 && string(argv[4]) == "p";

    fstream static_out, dynamic_out, guided_out;
    string threads_count_str = to_string(threads_count);
    static_out.open(path + "/omp_static_" + threads_count_str + EXTENSION, ios::app);
    dynamic_out.open(path + "/omp_dynamic_" + threads_count_str + EXTENSION, ios::app);
    guided_out.open(path + "/omp_guided_" + threads_count_str + EXTENSION, ios::app);

    log("Static, with q: %d\n", is_q);
    test_time(static_sum, static_out, is_q, is_print);
    log("Dynamic, with q: %d\n", is_q);
    test_time(dynamic_sum, dynamic_out, is_q, is_print);
    log("Guided, with q: %d\n", is_q);
    test_time(guided_sum, guided_out, is_q, is_print);

	static_out.close(); dynamic_out.close(); guided_out.close();

    return 0;
}

void log(const char* text, bool param)
{
    if (is_print)
    {
        printf(text, param);
    }
}
