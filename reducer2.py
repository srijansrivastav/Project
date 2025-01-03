#!/usr/bin/env python
""" reducer.py """
import sys

word = ""
current_word = ""
current_count = 0

# Read input (Map output) from STDIN
for line in sys.stdin:
  # Remove leading and trailing whitespace
  line = line.strip()
  # Parse input (map's tuples)
  word, count = line.split('\t', 1)
  # Convert count (currently a string) to int
  try:
    count = int(count)
  except ValueError:
    # Count is not a number; silently discard line
    continue

  # Hadoop sorts map output by key(word) before reduction
  # Word unchanged -> continue summation
  if current_word == word:
    current_count += count
  # New word encountered -> Write prev. and start new count
  else:
    if current_word:
      # Write result to STDOUT
      print '%s\t%s' % (current_word, current_count)
    current_count = count
    current_word = word

# Output final word if needed
if current_word == word:
  print '%s\t%s' % (current_word, current_count)
