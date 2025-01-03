#!/usr/bin/env python
""" reducer.py """
import sys

country = ""
current_country = ""
current_population = 0

# Read input (Map output) from STDIN
for line in sys.stdin:
  # Remove leading and trailing whitespace
  line = line.strip()
  # Parse input (map's tuples)
  country, population = line.split('\t', 1)
  # Convert count (currently a string) to int
  try:
    population = int(population)
  except ValueError:
    # Count is not a number; silently discard line
    continue

  # Hadoop sorts map output by key(word) before reduction
  # Word unchanged -> continue summation
  if current_country == country:
    current_population += population
  # New country encountered -> Write prev. and start new count
  else:
    if current_country:
      # Write result to STDOUT
      print '%s\t%s' % (current_country, current_population)
    current_population = population
    current_country = country

# Output final country if needed
if current_country == country:
  print '%s\t%s' % (current_country, current_population)
