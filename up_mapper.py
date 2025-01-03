#!/usr/bin/env python
""" mapper.py """
import sys
import re
# Read input from STDIN (standard input)
for line in sys.stdin:
  # Remove leading and trailing whitespace
  line = line.strip()
  # Split the line into records (Array of columns)
  cols = line.split('\t')
  # Process and output intermediate map results
  if re.match('^\\d{1,9}$', cols[4]):
    # Write the results to STDOUT (standard output)
    # Tab-delimited; Key-Value pair <country, city_population>
    print '%s\t%s' % (cols[2], cols[4])
    # Output here will be the input for the reduce

