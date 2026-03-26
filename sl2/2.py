# McCulloch-Pitts Neuron for ANDNOT function

def andnot_mcp(a, b):
    # weights
    w1 = 1     # weight for A
    w2 = -1    # weight for B (NOT effect)

    # threshold
    theta = 1

    # weighted sum
    net = a * w1 + b * w2

    # activation (step function)
    if net >= theta:
        return 1
    else:
        return 0


# Test all combinations
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("A B | Output")
for a, b in inputs:
    output = andnot_mcp(a, b)
    print(a, b, "|", output)