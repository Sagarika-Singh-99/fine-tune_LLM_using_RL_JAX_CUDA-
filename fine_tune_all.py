#Phase 1:

#select runtime as GPU
!nvidia-smi

!nvcc --version #need to know the version so you can download the exact cupy-cuda version

#install pyCUDA and CuPy
!pip install pycuda
!pip install cupy-cuda12x

#Write and Test a Basic CUDA Kernel using PyCUDA and Test CuPy for GPU Operations

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for adding two arrays
kernel_code = """
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
add_kernel = mod.get_function("add_kernel")

# Input data
N = 1000
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros_like(a)

# Allocate GPU memory
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Execute kernel on GPU
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
add_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

# Copy result back to CPU
cuda.memcpy_dtoh(c, c_gpu)

# Verify result
print("Result kernel (first 10 elements):", c[:10])
print("Expected (first 10 elements):", (a + b)[:10])

# Test CuPy for GPU Operations

import cupy as cp

# Create two random arrays on the GPU
a = cp.random.randn(1000, dtype=cp.float32)
b = cp.random.randn(1000, dtype=cp.float32)

# Add the arrays on the GPU
c = a + b

# Verify the result by transferring it to the CPU
print("Result CuPy (first 10 elements):", c[:10].get())

#Phase 2: 

# Optimizing the dot-product computation QK^T in the Multi-head attention mechanism layer:

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

#CUDA kernel for QK^T computation
kernel_code = """
__global__ void qk_dot_product(float *Q, float *K, float *output, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float dot_product = 0.0;
        for (int i = 0; i < d; ++i) {
            dot_product += Q[row * d + i] * K[col * d + i];
        }
        output[row * n + col] = dot_product;
    }
}
"""

mod = SourceModule(kernel_code)
qk_dot_product = mod.get_function("qk_dot_product")

# Input matrices Q and K
n, d = 128, 64  # Sequence length, embedding size
Q = np.random.randn(n, d).astype(np.float32)
K = np.random.randn(n, d).astype(np.float32)
output = np.zeros((n, n), dtype=np.float32)

# Allocate GPU memory
Q_gpu = cuda.mem_alloc(Q.nbytes)
K_gpu = cuda.mem_alloc(K.nbytes)
output_gpu = cuda.mem_alloc(output.nbytes)

# Copy data to GPU
cuda.memcpy_htod(Q_gpu, Q)
cuda.memcpy_htod(K_gpu, K)

# Define grid and block dimensions
block_dim = (16, 16, 1)
grid_dim = ((n + block_dim[0] - 1) // block_dim[0], (n + block_dim[1] - 1) // block_dim[1])

# Launch kernel
qk_dot_product(Q_gpu, K_gpu, output_gpu, np.int32(n), np.int32(d), block=block_dim, grid=grid_dim)

# Copy result back to CPU
cuda.memcpy_dtoh(output, output_gpu)

# Verify result
print("GPU Output (first 5x5):")
print(output[:5, :5])
print("Expected (first 5x5):")
print(np.dot(Q, K.T)[:5, :5])

#Phase 3: 

#Optimize Softmax Layer

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for softmax
kernel_code = """
__global__ void softmax(float *input, float *output, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Step 1: Find the maximum value in the row (for numerical stability)
        float max_val = -1e20;
        for (int i = 0; i < d; ++i) {
            max_val = fmaxf(max_val, input[row * d + i]);
        }

        // Step 2: Compute exponentials and sum them
        float sum = 0.0;
        for (int i = 0; i < d; ++i) {
            float exp_val = expf(input[row * d + i] - max_val);
            sum += exp_val;
            output[row * d + i] = exp_val;  // Temporarily store exponentials
        }

        // Step 3: Normalize each element by the sum
        for (int i = 0; i < d; ++i) {
            output[row * d + i] /= sum;
        }
    }
}
"""

mod = SourceModule(kernel_code)
softmax = mod.get_function("softmax")

# Input matrix (QK^T) from the previous step
n, d = 128, 128  # Rows and columns (sequence length)
input_matrix = np.random.randn(n, d).astype(np.float32)
output_matrix = np.zeros_like(input_matrix)

# Allocate GPU memory
input_gpu = cuda.mem_alloc(input_matrix.nbytes)
output_gpu = cuda.mem_alloc(output_matrix.nbytes)

# Copy input matrix to GPU
cuda.memcpy_htod(input_gpu, input_matrix)

# Define grid and block dimensions
block_dim = 32  # Each block processes one row
grid_dim = (n + block_dim - 1) // block_dim

# Launch softmax kernel
softmax(input_gpu, output_gpu, np.int32(n), np.int32(d), block=(block_dim, 1, 1), grid=(grid_dim, 1))

# Copy result back to CPU
cuda.memcpy_dtoh(output_matrix, output_gpu)

# Verify results by comparing with NumPy softmax
def numpy_softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

expected_output = numpy_softmax(input_matrix)

# Compare first 5 rows
print("GPU Output (first 5 rows):")
print(output_matrix[:5, :5])
print("Expected Output (first 5 rows):")
print(expected_output[:5, :5])

# Verify correctness
assert np.allclose(output_matrix, expected_output, atol=1e-5), "Softmax outputs do not match!"

#Phase 4:

#code for integration - 

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Dimensions
n, d = 128, 64  # Sequence length (n), embedding size (d)
dk = d  # Dimension of keys

# Initialize Q, K, V matrices
Q = np.random.randn(n, d).astype(np.float32)  # Query matrix
K = np.random.randn(n, d).astype(np.float32)  # Key matrix
V = np.random.randn(n, d).astype(np.float32)  # Value matrix
output_attention = np.zeros_like(Q)  # Placeholder for attention output

# Allocate GPU memory
Q_gpu = cuda.mem_alloc(Q.nbytes)
K_gpu = cuda.mem_alloc(K.nbytes)
V_gpu = cuda.mem_alloc(V.nbytes)
QK_gpu = cuda.mem_alloc(n * n * np.dtype(np.float32).itemsize)  # QK^T result
softmax_output_gpu = cuda.mem_alloc(n * n * np.dtype(np.float32).itemsize)  # Softmax output
output_gpu = cuda.mem_alloc(output_attention.nbytes)

# Copy data to GPU
cuda.memcpy_htod(Q_gpu, Q)
cuda.memcpy_htod(K_gpu, K)
cuda.memcpy_htod(V_gpu, V)

# CUDA kernels: QK^T and Softmax
kernel_code = """
__global__ void qk_dot_product(float *Q, float *K, float *output, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float dot_product = 0.0;
        for (int i = 0; i < d; ++i) {
            dot_product += Q[row * d + i] * K[col * d + i];
        }
        output[row * n + col] = dot_product;
    }
}

__global__ void softmax(float *input, float *output, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Step 1: Find the maximum value in the row (for numerical stability)
        float max_val = -1e20;
        for (int i = 0; i < d; ++i) {
            max_val = fmaxf(max_val, input[row * d + i]);
        }

        // Step 2: Compute exponentials and sum them
        float sum = 0.0;
        for (int i = 0; i < d; ++i) {
            float exp_val = expf(input[row * d + i] - max_val);
            sum += exp_val;
            output[row * d + i] = exp_val;  // Temporarily store exponentials
        }

        // Step 3: Normalize each element by the sum
        for (int i = 0; i < d; ++i) {
            output[row * d + i] /= sum;
        }
    }
}
"""

# Compile the kernels
mod = SourceModule(kernel_code)
qk_dot_product = mod.get_function("qk_dot_product")
softmax = mod.get_function("softmax")

# Define grid and block dimensions
block_dim = (16, 16, 1)
grid_dim_qk = ((n + block_dim[0] - 1) // block_dim[0], (n + block_dim[1] - 1) // block_dim[1])
grid_dim_softmax = (n + 31) // 32  # For row-wise processing

# Step 1: Launch QK^T kernel
qk_dot_product(
    Q_gpu, K_gpu, QK_gpu, 
    np.int32(n), np.int32(d),
    block=block_dim, grid=grid_dim_qk
)

# Step 2: Launch Softmax kernel
softmax(
    QK_gpu, softmax_output_gpu, 
    np.int32(n), np.int32(n),
    block=(32, 1, 1), grid=(grid_dim_softmax, 1)
)

# Copy softmax output back to CPU for verification
softmax_output = np.empty((n, n), dtype=np.float32)
cuda.memcpy_dtoh(softmax_output, softmax_output_gpu)

# Step 3: Compute Attention Output (Softmax * V)
# This step can also be optimized with CUDA if needed
output_attention = np.dot(softmax_output, V)

# Verify results
print("Softmax Output (first 5 rows):")
print(softmax_output[:5, :5])
print("Attention Output (first 5 rows):")
print(output_attention[:5, :5])

# CUDA kernel for Softmax(QK^T) * V
kernel_code += """
__global__ void softmax_v_mul(float *softmax, float *V, float *output, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < d) {
        float value = 0.0;
        for (int i = 0; i < n; ++i) { // Iterate over the sequence length
            value += softmax[row * n + i] * V[i * d + col];
        }
        output[row * d + col] = value;
    }
}
"""

mod = SourceModule(kernel_code)
softmax_v_mul = mod.get_function("softmax_v_mul")

# Allocate memory for the final attention output on the GPU
output_attention_gpu = cuda.mem_alloc(output_attention.nbytes)

# Define grid and block dimensions for matrix multiplication
block_dim = (16, 16, 1)
grid_dim = ((n + block_dim[0] - 1) // block_dim[0], (d + block_dim[1] - 1) // block_dim[1])

# Launch the kernel to compute Softmax(QK^T) * V
softmax_v_mul(
    softmax_output_gpu,  # Softmax output on GPU
    V_gpu,               # Value matrix on GPU
    output_attention_gpu, # Final attention output on GPU
    np.int32(n),         # Sequence length
    np.int32(d),         # Embedding dimension
    block=block_dim,
    grid=grid_dim
)

# Copy the final attention output back to the CPU for verification
cuda.memcpy_dtoh(output_attention, output_attention_gpu)

# Verify the output
print("Optimized Attention Output (first 5 rows):")
print(output_attention[:5, :5])

# Phase 5:

# fine tune - 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.optim import AdamW  # Use PyTorch's AdamW optimizer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

# Load GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Fix padding token issues
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Prompts for text generation
prompts = ["Once upon a time", "The future of AI", "In a distant galaxy"]

# Define reward function
def compute_rewards(predictions, prompts):
    """Enhanced reward function with fluency, relevance, and diversity."""
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Use a sentence embedding model
    rewards = []

    for pred, prompt in zip(predictions, prompts):
        # Fluency (proxy using BLEU score or perplexity from another model)
        fluency_score = sentence_bleu([["sample", "reference"]], pred.split())

        # Relevance (cosine similarity between prompt and generated text embeddings)
        prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
        pred_embedding = embedding_model.encode(pred, convert_to_tensor=True)
        relevance_score = util.cos_sim(prompt_embedding, pred_embedding).item()

        # Diversity (e.g., percentage of unique bigrams)
        tokens = pred.split()
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        diversity_score = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0

        # Final reward: weighted combination
        reward = fluency_score + 0.7 * relevance_score + 0.5 * diversity_score
        rewards.append(reward)

    return rewards

# RL fine-tuning loop
for epoch in range(3):  # Set the number of epochs as required
    total_loss = 0
    all_rewards = []

    for prompt in prompts:
        # Step 1: Generate outputs with attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Add attention mask here
            max_length=50,
            do_sample=True,
        )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")

        # Step 2: Compute rewards
        rewards = compute_rewards([generated_text], [prompt])  # Pass both generated_text and corresponding prompt
        all_rewards.append(rewards[0])  # Collect reward for this output

        # Step 3: Compute policy loss for RL
        # Re-run the forward pass to get logits for the generated tokens
        combined_inputs = torch.cat([inputs["input_ids"], outputs], dim=-1)  # Combine input and generated tokens
        logits = model(combined_inputs).logits  # Get logits for combined sequence
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract logits corresponding to the generated tokens
        generated_logits = log_probs[:, -outputs.shape[1]:, :]  # Extract logits for generated tokens
        generated_tokens = outputs  # Tokens generated by the model

        # Select log probabilities of generated tokens
        selected_log_probs = torch.gather(
            generated_logits, 2, generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Compute reward-weighted policy loss
        advantages = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        loss = -torch.mean(advantages * selected_log_probs.sum(dim=1))

        # Step 4: Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    # Print average reward and loss for the epoch
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"Epoch {epoch + 1}: Average Reward = {avg_reward:.4f}, Total Loss = {total_loss:.4f}")

# Save fine-tuned model
model.save_pretrained("./fine_tuned_gpt2_rl")
tokenizer.save_pretrained("./fine_tuned_gpt2_rl")

# Phase 6:

#test prompts - 

test_prompts = [
    "A world without technology",
    "Exploring the depths of the ocean",
    "The history of the human race",
]
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicitly pass attention mask
        max_length=50,
        do_sample=True,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")

# Phase 7: 

#deploy model

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_gpt2_rl").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_gpt2_rl")

# Generate new text
prompt = "The mysteries of space"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  # Explicitly pass attention mask
    max_length=50,
    do_sample=True,
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")



