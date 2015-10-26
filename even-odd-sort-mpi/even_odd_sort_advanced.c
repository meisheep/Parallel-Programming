#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define true 1
#define false 0
#define MASTER_RANK 0
#define MSG_RECV 0

void swap(int *a, int *b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
	return;
}

void seq_forward_bubble_sort(int* arr, int length) {
	int i;
	for(i = 0;i < length - 1;i++) {
		if(arr[i] > arr[i+1]) {
			swap(&arr[i], &arr[i+1]);
		} else {
			return;
		}
	}
}

void seq_backward_bubble_sort(int* arr, int length) {
	int i;
	for(i = length - 1;i > 0;i--) {
		if(arr[i] < arr[i-1]) {
			swap(&arr[i], &arr[i-1]);
		} else {
			return;
		}
	}
}

int compare(const void* a, const void* b) {
	if (*(signed int*)a > *(signed int*)b)
		return 1;
	else if (*(signed int*)a < *(signed int*)b)
		return -1;
	else
		return 0;
}

int main (int argc, char *argv[]) {
	int rank, size;

	MPI_File fh_in, fh_out;
	MPI_Offset offset;
	MPI_Status status;
	MPI_Group origin_group, new_group;
	MPI_Comm custom_world = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(custom_world, &size);
	MPI_Comm_rank(custom_world, &rank);

	// read command
	if (argc < 4) {
		if (rank == MASTER_RANK) {
			fprintf(stderr, "Insufficient args\n");
			fprintf(stderr, "Usage: %s N input_file output_file", argv[0]);
		}
		return 0;
	}

	const int N = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];

	// Deal with the case where (N < size)
	if (N < size) {
		// obtain the group of proc. in the world communicator
		MPI_Comm_group(custom_world, &origin_group);

		// remove unwanted ranks
		int ranges[][3] = {{N, size-1, 1}};
		MPI_Group_range_excl(origin_group, 1, ranges, &new_group);

		// create a new communicator
		MPI_Comm_create(custom_world, new_group, &custom_world);

		if (custom_world == MPI_COMM_NULL) {
			// terminate those unwanted processes
			MPI_Finalize();
			exit(0);
		}

		size = N;
	}

	// Read file using MPI-IO
	int *local_buf;
	int num_per_node = N / size;
	offset = rank * num_per_node * sizeof(int);

	if (rank == (size - 1)) {
		num_per_node += N % size;
	}

	local_buf = malloc(num_per_node * sizeof(int));

	MPI_File_open(custom_world, INPUT_NAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);
	MPI_File_read_at(fh_in, offset, local_buf, num_per_node, MPI_INT, &status);
	MPI_File_close(&fh_in);

	// local sorting once
	qsort(local_buf, num_per_node, sizeof(int), compare);

	// Odd-even sort
	int sorted = false, all_sorted = false;
	int recv;
	while (!sorted || !all_sorted) {
		sorted = true;

		// transportation
		// odd phase
		if (rank % 2) {
			MPI_Send(&local_buf[0], 1, MPI_INT, rank - 1, MSG_RECV, custom_world);
			MPI_Recv(&recv, 1, MPI_INT, rank - 1, MSG_RECV, custom_world, &status);
			if (recv > local_buf[0]) {
				local_buf[0] = recv;
				seq_forward_bubble_sort(local_buf, num_per_node);
				sorted = false;
			}
		} else if (rank != (size - 1)) {
			MPI_Recv(&recv, 1, MPI_INT, rank + 1, MSG_RECV, custom_world, &status);
			if(recv < local_buf[num_per_node - 1]) {
				swap(&recv, &local_buf[num_per_node - 1]);
				seq_backward_bubble_sort(local_buf, num_per_node);
				sorted = false;
			}
			MPI_Send(&recv, 1, MPI_INT, rank + 1, MSG_RECV, custom_world);
		}

		// even phase
		if ((rank % 2) == 0 && rank != MASTER_RANK) {
			MPI_Send(&local_buf[0], 1, MPI_INT, rank - 1, MSG_RECV, custom_world);
			MPI_Recv(&recv, 1, MPI_INT, rank - 1, MSG_RECV, custom_world, &status);
			if (recv > local_buf[0]) {
				local_buf[0] = recv;
				seq_forward_bubble_sort(local_buf, num_per_node);
				sorted = false;
			}
		} else if(rank > MASTER_RANK && rank != (size - 1)) {
			MPI_Recv(&recv, 1, MPI_INT, rank + 1, MSG_RECV, custom_world, &status);
			if(recv < local_buf[num_per_node - 1]) {
				swap(&recv, &local_buf[num_per_node - 1]);
				seq_backward_bubble_sort(local_buf, num_per_node);
				sorted = false;
			}
			MPI_Send(&recv, 1, MPI_INT, rank + 1, MSG_RECV, custom_world);
		}

		MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_LAND, custom_world);
	}

	// Write file using MPI-IO
	MPI_File_open(custom_world, OUTPUT_NAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);
	MPI_File_write_at(fh_out, offset, local_buf, num_per_node, MPI_INT, &status);
	MPI_File_close(&fh_out);

	free(local_buf);

	MPI_Barrier(custom_world);
	MPI_Finalize();

	return 0;
}
