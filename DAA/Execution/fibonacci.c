#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long fibonacci_iterative(long long n){
    if (n == 0) return 0;
    if (n == 1) return 1;
    long long a = 0, b = 1, temp = 0;
    for(long long i = 2; i <= n; i++){
        temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

long long fibonacci_recursive(long long n){
    if(n <= 1) return n;
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2);
}

void printElapsed(clock_t start, clock_t end){
    double seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    double ms = seconds * 1000.0;
    printf("Elapsed Time: %.4f ms\n", ms);
}

int main(void){
    int n = -1, choice = -1;
    long long result = 0;

    printf("Enter Value for n: ");
    if (scanf("%d", &n) != 1){
        fprintf(stderr, "Invalid input\n");
        return 1;
    }
    if(n < 0){
        printf("n should be non-negative\n");
        return 1;
    }

    printf("\nChoose compute technique:\n1) Iterative Method.\n2) Recursive Technique.\n: ");
    if (scanf("%d", &choice) != 1){
        fprintf(stderr, "Invalid input\n");
        return 1;
    }
    printf("You chose: %d\n", choice);

    if(choice == 2 && n > 40){
        /* consume leftover newline before reading confirmation */
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF) { /* flush */ }
        printf("Time complexity of recursive algorithm is exponential, so computation may take a long time\nDo you wish to continue (y/n): ");
        int c = getchar();
        /* skip additional characters until newline */
        while (c != '\n' && getchar() != '\n') { /* flush rest of line */ }

        if (c != 'y' && c != 'Y') {
            printf("Aborted Recursion\n");
            return 1;
        }
    }

    clock_t start, end;
    if(choice == 1){
        start = clock();
        result = fibonacci_iterative(n);
        end = clock();
        printf("\nIterative Fibonacci(%d): %lld\n", n, result);
        printElapsed(start, end);
    }
    else if(choice == 2){
        start = clock();
        result = fibonacci_recursive(n);
        end = clock();
        printf("\nRecursive Fibonacci(%d): %lld\n", n, result);
        printElapsed(start, end);
    }
    else{
        printf("Invalid Choice\n");
        return 1;
    }
    return 0;
}
