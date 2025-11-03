#include <stdio.h>
#include <stdlib.h>

long long steps = 0;  // Global counter to track algorithmic steps

// Structure to represent a node in the Huffman Tree
typedef struct Node {
    char data;           // Character (symbol)
    int freq;            // Frequency of the character
    struct Node *left;   // Left child
    struct Node *right;  // Right child
} Node;

// Function to create a new tree node
Node* newNode(char data, int freq) {
    steps++;
    Node* node = (Node*)malloc(sizeof(Node));  // Dynamically allocate memory
    node->data = data;
    node->freq = freq;
    node->left = node->right = NULL;           // Initially no children
    return node;
}

// Utility function to swap two node pointers (used in heap operations)
void swap(Node** a, Node** b) {
    steps++;
    Node* temp = *a;
    *a = *b;
    *b = temp;
}

// Function to maintain min-heap property (heapify at index i)
void heapify(Node** arr, int n, int i) {
    steps++;
    int smallest = i;
    int left = 2 * i + 1;   // Index of left child
    int right = 2 * i + 2;  // Index of right child

    // Compare left child with current smallest
    if (left < n && arr[left]->freq < arr[smallest]->freq)
        smallest = left;

    // Compare right child with current smallest
    if (right < n && arr[right]->freq < arr[smallest]->freq)
        smallest = right;

    // If the smallest is not the current index, swap and recurse
    if (smallest != i) {
        swap(&arr[i], &arr[smallest]);
        heapify(arr, n, smallest);
    }
}

// Build a min-heap from the array
void buildHeap(Node** arr, int n) {
    // Start from last non-leaf node and heapify upwards
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
}

// Extract (remove and return) the node with minimum frequency
Node* extractMin(Node** arr, int* n) {
    steps++;
    Node* minNode = arr[0];             // Root (minimum element)
    arr[0] = arr[--(*n)];               // Move last node to root
    heapify(arr, *n, 0);                // Restore min-heap property
    return minNode;
}

// Insert a new node into the min-heap
void insertHeap(Node** arr, int* n, Node* node) {
    steps++;
    int i = (*n)++;
    arr[i] = node;

    // Move the new node up to maintain heap order
    while (i && arr[i]->freq < arr[(i - 1) / 2]->freq) {
        swap(&arr[i], &arr[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

// Check if a node is a leaf (no children)
int isLeaf(Node* root) {
    steps++;
    return !(root->left) && !(root->right);
}

// Recursive function to print Huffman codes from the tree
void printCodes(Node* root, int arr[], int top) {
    if (root->left) {               // Add '0' for left edge
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }

    if (root->right) {              // Add '1' for right edge
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }

    // If the node is a leaf, print the character and its Huffman code
    if (isLeaf(root)) {
        printf("%c: ", root->data);
        for (int i = 0; i < top; i++)
            printf("%d", arr[i]);
        printf("\n");
    }
}

// Main Huffman building function (Greedy Method)
Node* buildHuffman(char data[], int freq[], int n) {
    steps++;
    Node* arr[100];   // Array to act as heap (max 100 symbols)

    // Create initial nodes for all characters
    for (int i = 0; i < n; i++)
        arr[i] = newNode(data[i], freq[i]);

    // Convert array into a min-heap
    buildHeap(arr, n);

    // Combine the two smallest nodes until one node remains
    while (n > 1) {
        Node* left = extractMin(arr, &n);
        Node* right = extractMin(arr, &n);

        // Create a new internal node with frequency sum
        Node* top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;

        insertHeap(arr, &n, top);   // Insert new node into heap
    }

    return arr[0];  // The remaining node is the root of the Huffman Tree
}

int main() {
    // Example data and frequency
    char data[] = {'a', 'b', 'c', 'd', 'e', 'f'};
    int freq[] = {5, 9, 12, 13, 16, 45};
    int n = sizeof(data) / sizeof(data[0]);

    // Build Huffman Tree
    Node* root = buildHuffman(data, freq, n);

    // Print the Huffman Codes
    int code[100];
    printf("Huffman Codes:\n");
    printCodes(root, code, 0);

    // Print analysis info
    printf("\nStep Count = %lld\n", steps);
    printf("Time Complexity: O(n log n)\n");
    printf("Space Complexity: O(n)\n");

    return 0;
}
