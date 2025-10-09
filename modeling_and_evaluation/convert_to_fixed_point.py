# Assume a Q15 fixed-point format (15 fractional bits)
fractional_bits = 15

# Floating-point number to convert
float_value = 0.00235528545

# Scale the floating-point number
scaled_value = float_value * (2 ** fractional_bits)

# Convert to integer (truncating)
fixed_point_integer = int(scaled_value)

print(f"Floating-point value: {float_value}")
print(f"Scaled value: {scaled_value}")
print(f"Fixed-point integer (Q15): {fixed_point_integer}")