import random

def check(samples, m):
    dis = [[0 for i in range(m)] for j in range(m)]

    for i in range(m):
        for j in range(i):
            dis[i][j] = dis[j][i] = sum(map(lambda x, y : (x - y) ** 2, samples[i], samples[j]))

    for i in range(m):
        if len(set(dis[i])) != len(dis[i]):
            return False

    return True

def generate_sample(m, n, k, output_file):
    samples = [[str(random.randint(-10, 10)) for i in range(n)] for j in range(m)]
    print 'now checking sample...'
    if check(samples, m):
        print 'pass'
    else:
        print 'failed: try again'
        return generate_sample(m, n, k, output_file)

    print 'write to file: %s' % output_file
    with open(output_file, 'w') as fout:
        fout.write('%i %i %i\n' % (m, n, k))
        for sample in samples:
            fout.write('%s\n' % ' '.join(sample))

print 'small1...'
generate_sample(1024, 256, 8, 'small1.in')
print 'small2...'
generate_sample(1024, 256, 8, 'small2.in')
print 'small3...'
generate_sample(1024, 256, 8, 'small3.in')
print 'middle1...'
generate_sample(4096, 1024, 16, 'middle1.in')
print 'middle2...'
generate_sample(4096, 1024, 16, 'middle2.in')
print 'middle2...'
generate_sample(4096, 1024, 16, 'middle3.in')
print 'large1...'
generate_sample(16384, 4096, 32, 'large1.in')
print 'large2...'
generate_sample(16384, 4096, 32, 'large2.in')
print 'large3...'
generate_sample(16384, 4096, 32, 'large3.in')
