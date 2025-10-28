#include <stdio.h>

// Function: Non-recursive (Iterative) Fibonacci
int fibonacci_iterative(int n) {
    if (n <= 1)
        return n;

    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Function: Recursive Fibonacci
int fibonacci_recursive(int n) {
    if (n <= 1)
        return n;
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2);
}

// Main function
int main() {
    int n, choice;
    printf("Enter the value of n: ");
    scanf("%d", &n);

    printf("\nChoose method:\n");
    printf("1. Iterative (Non-Recursive)\n");
    printf("2. Recursive\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    int result;
    if (choice == 1) {
        result = fibonacci_iterative(n);
        printf("\n[Iterative] Fibonacci(%d) = %d\n", n, result);
        printf("Time Complexity: O(n)\n");
        printf("Space Complexity: O(1)\n");
    } 
    else if (choice == 2) {
        result = fibonacci_recursive(n);
        printf("\n[Recursive] Fibonacci(%d) = %d\n", n, result);
        printf("Time Complexity: O(2^n)\n");
        printf("Space Complexity: O(n)\n");
    } 
    else {
        printf("\nInvalid choice!\n");
    }

    return 0;
}
