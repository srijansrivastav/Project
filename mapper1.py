#!/usr/bin/env python
""" mapper.py """
import sys
# Read input from STDIN (standard input)
for line in sys.stdin:
  # Remove leading and trailing whitespace
  line = line.strip()
  # Split the line into tokens (words)
  words = line.split()
  # Process and output intermediate map results
  for word in words:
    # Write the results to STDOUT (standard output)
    # Tab-delimited; Key-Value pair <word, 1>
    print '%s\t%s' % (word, 1)
    # Output here will be the input for the reduce

