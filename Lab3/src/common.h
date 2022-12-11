// common.h
#define Q_PARAM 22

double* create_double_array(long size)
{
	double* array = (double*) malloc(sizeof(double) * size);
	for (int j = 0; j < size; ++j)
	{
		array[j] = j;
	}
	return array;
}

double sum_with_q(double arg) {
	double result = 0;
	double normal_arg = arg / Q_PARAM;
	for (int k = 0; k < Q_PARAM; ++k)
	{
		result += normal_arg;
	}
	return result;
}

