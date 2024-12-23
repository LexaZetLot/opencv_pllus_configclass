class PLY:
    def __init__(self, dirSaveFilePly):
        self.dirSaveFilePly = dirSaveFilePly

    def save(self, structure):
        structure /= 2
        fp = open(self.dirSaveFilePly, 'w')
        fp.write('ply\n')
        fp.write('format ascii 1.0\n')
        fp.write('element vertex %d\n' % len(structure.T[0]))
        fp.write('property float x\n')
        fp.write('property float y\n')
        fp.write('property float z\n')
        fp.write('end_header\n')

        for element in zip(structure.T[0], structure.T[1], structure.T[2]):
            x, y, z = element
            fp.write('%f %f %f\n' % (x, y, z))