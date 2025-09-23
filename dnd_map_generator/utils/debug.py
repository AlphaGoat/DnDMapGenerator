def print_sizes(model, input_tensor):
    output = input_tensor
    print(f"{'Layer':30} {'Output Shape'}")
    for m in model.children():
        output = m(output)
        print(f"{m.__class__.__name__:30} {list(output.shape)}")
    return output