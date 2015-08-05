import random

R = 40

def check(samples, m, k):
    dis = [[1073741824 for i in range(m)] for j in range(m)]
    answers = []

    for i in range(m):
        for j in range(i):
            dis[i][j] = dis[j][i] = sum(map(lambda x, y : (x - y) ** 2, samples[i], samples[j]))

    for i in range(m):
        distance = sorted(dis[i])[:(k + 1)]
        if len(distance) != len(set(distance)):
            return False, []

        idx = [j for j in range(m)]
        answers.append(sorted(idx, key=lambda x : dis[i][x])[:k])

    return True, answers

def generate_sample(m, n, k, output_file):
    samples = [[random.randint(-R, R) for i in range(n)] for j in range(m)]
    print 'now checking sample...'
    ret, answers = check(samples, m, k)
    if ret:
        print 'pass'
    else:
        print 'failed: try again'
        return generate_sample(m, n, k, output_file)

    print 'write to in file: %s.in' % output_file
    with open('%s.in' % output_file, 'w') as fout:
        fout.write('%i %i %i\n' % (m, n, k))
        for sample in samples:
            fout.write('%s\n' % ' '.join(map(str, sample)))

    print 'write to answer file: %s.ans' % output_file
    with open('%s.ans' % output_file, 'w') as fout:
        for answer in answers:
            fout.write('%s \n' % ' '.join(map(str, answer)))

print 'small1...'
generate_sample(1024, 256, 8, 'small1')
print 'small2...'
generate_sample(1024, 256, 8, 'small2')
print 'small3...'
generate_sample(1024, 256, 8, 'small3')
print 'middle1...'
generate_sample(4096, 1024, 16, 'middle1')
print 'middle2...'
generate_sample(4096, 1024, 16, 'middle2')
print 'middle2...'
generate_sample(4096, 1024, 16, 'middle3')
print 'large1...'
generate_sample(16384, 4096, 32, 'large1')
print 'large2...'
generate_sample(16384, 4096, 32, 'large2')
print 'large3...'
generate_sample(16384, 4096, 32, 'large3')
