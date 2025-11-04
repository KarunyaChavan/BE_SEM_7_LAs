#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long long steps = 0;  // Step counter

// Node structure for Huffman tree
typedef struct Node {
    char data;
    int freq;
    struct Node *left, *right;
} Node;

// Create new node
Node* newNode(char data, int freq) {
    steps++;
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->freq = freq;
    node->left = node->right = NULL;
    return node;
}

// Swap helper for heap
void swap(Node** a, Node** b) {
    Node* temp = *a;
    *a = *b;
    *b = temp;
}

// Maintain min-heap property
void heapify(Node** arr, int n, int i) {
    steps++;
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left]->freq < arr[smallest]->freq)
        smallest = left;
    if (right < n && arr[right]->freq < arr[smallest]->freq)
        smallest = right;

    if (smallest != i) {
        swap(&arr[i], &arr[smallest]);
        heapify(arr, n, smallest);
    }
}

// Build heap from array
void buildHeap(Node** arr, int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
}

// Extract minimum frequency node
Node* extractMin(Node** arr, int* n) {
    steps++;
    Node* minNode = arr[0];
    arr[0] = arr[--(*n)];
    heapify(arr, *n, 0);
    return minNode;
}

// Insert a node into heap
void insertHeap(Node** arr, int* n, Node* node) {
    steps++;
    int i = (*n)++;
    arr[i] = node;
    while (i && arr[i]->freq < arr[(i - 1) / 2]->freq) {
        swap(&arr[i], &arr[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

// Check if a node is a leaf
int isLeaf(Node* root) {
    return root && !root->left && !root->right;
}

// Print Huffman codes recursively
void printCodes(Node* root, int arr[], int top) {
    if (!root) return;

    if (root->left) {
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }
    if (root->right) {
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }
    if (isLeaf(root)) {
        printf("%c: ", root->data);
        for (int i = 0; i < top; i++) printf("%d", arr[i]);
        printf("\n");
    }
}

// Build Huffman tree
Node* buildHuffman(char data[], int freq[], int n) {
    Node* arr[256];
    for (int i = 0; i < n; i++)
        arr[i] = newNode(data[i], freq[i]);
    buildHeap(arr, n);

    while (n > 1) {
        Node* left = extractMin(arr, &n);
        Node* right = extractMin(arr, &n);
        Node* top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertHeap(arr, &n, top);
    }
    return arr[0];
}

// Automatically calculate frequencies
int calculateFrequency(const char* str, char data[], int freq[]) {
    int count[256] = {0};
    for (int i = 0; str[i] != '\0'; i++) {
        unsigned char ch = str[i];
        count[ch]++;
    }

    int n = 0;
    for (int i = 0; i < 256; i++) {
        if (count[i] > 0) {
            data[n] = (char)i;
            freq[n] = count[i];
            n++;
        }
    }
    return n;
}

int main() {
    char input[1000];
    printf("Enter a string: ");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';  // Remove newline

    char data[256];
    int freq[256];
    int n = calculateFrequency(input, data, freq);

    printf("\nCharacter Frequencies:\n");
    for (int i = 0; i < n; i++)
        printf("%c: %d\n", data[i], freq[i]);

    Node* root = buildHuffman(data, freq, n);

    int code[256];
    printf("\nHuffman Codes:\n");
    printCodes(root, code, 0);

    printf("\nStep Count = %lld\n", steps);
    printf("Time Complexity: O(n log n)\n");
    printf("Space Complexity: O(n)\n");
    return 0;
}
