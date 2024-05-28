import tritonclient.grpc as grpcclient
import numpy as np

model_name = "ddnet_jhmdb"
input_name_0 = "input__0"
input_name_1 = "input__1"
output_name = "output__0"

# Create an instance of the InferenceServerClient
client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Create input data
input_data_0 = np.random.randn(512, 32, 105).astype(np.float32)
input_data_1 = np.random.randn(512, 32, 15, 2).astype(np.float32)

# Create input tensors
input_tensor_0 = grpcclient.InferInput(input_name_0, input_data_0.shape, "FP32")
input_tensor_1 = grpcclient.InferInput(input_name_1, input_data_1.shape, "FP32")

# Set the input data
input_tensor_0.set_data_from_numpy(input_data_0)
input_tensor_1.set_data_from_numpy(input_data_1)

# Create output tensor
output_tensor = grpcclient.InferRequestedOutput(output_name)

# Run inference
results = client.infer(model_name=model_name, inputs=[input_tensor_0, input_tensor_1], outputs=[output_tensor])

# Get results
output_data = results.as_numpy(output_name)
print("Output:", output_data)
