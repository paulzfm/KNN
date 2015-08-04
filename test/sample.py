import random

def generate_sample(m, n, k, output_file):
    samples = [[str(random.randint(-10, 10)) for i in range(n)] for j in range(m)]
    with open(output_file, 'w') as fout:
        fout.write('%i %i %i\n' % (m, n, k))
        for sample in samples:
            fout.write('%s\n' % ' '.join(sample))

generate_sample(1024, 256, 8, 'small.in')
generate_sample(4096, 1024, 16, 'middle.in')
generate_sample(16384, 4096, 32, 'large.in')
