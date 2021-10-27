ml_default_column = set(['FZI', 'GK', 'BK', 'RHOB', 'IK', 'DT', 'NGK', 'MD'])
gis_skipped = set(['MD'])
zolushka = ml_default_column.difference(gis_skipped)


print(zolushka)

