// common.h
#define Q_PARAM 22
#define TYPE_PARAM double
#define MPI_TYPE MPI_DOUBLE
#define NMAX 6000000
#define SCATTERV_PARAM 11

TYPE_PARAM* create_array(int N)
{
    return (TYPE_PARAM*)malloc(N * sizeof(TYPE_PARAM));
}

TYPE_PARAM* init_array(TYPE_PARAM* array, int size)
{
	for (int j = 0; j < size; ++j)
	{
		array[j] = j;
	}
	return array;
}

void delete_array(TYPE_PARAM* array)
{
    free(array);
}

double sum_with_q(TYPE_PARAM left, TYPE_PARAM right)
{
    TYPE_PARAM result = 0;
    TYPE_PARAM normal_left = left / Q_PARAM;
    TYPE_PARAM normal_right = right / Q_PARAM;
	for (int k = 0; k < Q_PARAM; ++k)
	{
		result += normal_left + normal_right;
	}
	return result;
}

TYPE_PARAM* sum(TYPE_PARAM* destination, const TYPE_PARAM* left_array, const TYPE_PARAM* right_array, long n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        destination[i] = left_array[i] + right_array[i];
    }
    return destination;
}

TYPE_PARAM* sum_q(TYPE_PARAM* destination, const TYPE_PARAM* left_array, const TYPE_PARAM* right_array, long n)
{
    int i;
    for (i = 0; i < NMAX; i++)
    {
        destination[i] += sum_with_q(left_array[i], right_array[i]);
    }
    return destination;
}

void print_array(TYPE_PARAM* array, int size)
{
    int half_max = 15;
    int center_border = size > half_max ? half_max : size;
    printf("[ ");
    for (int i = 0; i < center_border; ++i)
    {
        printf("%10.2f ", array[i]);
    }
    if (size > 2 * half_max)
    {
        printf("... ");
    }
    printf("... ");
    for (int i = size - 15; i < size; ++i)
    {
        printf("%10.2f ", array[i]);
    }
    printf("]\n");
}
