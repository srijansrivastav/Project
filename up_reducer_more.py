#!/usr/bin python
import sys
current_country = ""
current_population_max = 0
country = ""
city = ""

for line in sys.stdin:
	country, city_population = line.split('\t', 1)
	city_name, population = city_population.split(',', 1)

	try:
		population = int(population)
	except ValueError:
		continue

	if current_country == country:
		if population > current_population_max:
			current_population_max = population
			city = city_name

	else:
		if current_country:
			print '%s\t%s,%s' % (current_country, city, current_population_max)

		current_population_max = population
		current_country = country

if current_country == country:
	print '%s\t%s,%d' % (current_country, city, current_population_max)
