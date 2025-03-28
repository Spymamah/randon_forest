def square_point(x, y, z):
  x_squared = x * x
  y_squared = y * y
  z_squared = z * z
  # Return all three values:
  print(x_squared, y_squared, z_squared)
  return x_squared, y_squared, z_squared

three_squared, four_squared, five_squared = square_point(3, 4, 5)