import numpy as np

def conv_flops(H, W, C, F, L=3):
    return L**2 * F * (H * W * C)

def scat_flops(H, W, C):
    return H*W*C*36

def scatmix_flops(H, W, C):
    return (7/4 * 7*C + 36) * H * W * C

def main():
    # ScatNet A
    C = 96
    layer1 = scat_flops(32, 32, 3)
    layer2 = scat_flops(16, 16, 21)
    conv2_1 = conv_flops(8, 8, 149, 2*C)
    conv2_2 = conv_flops(8, 8, 2*C, 2*C)
    conv3_1 = conv_flops(8, 8, 2*C, 4*C)
    conv3_2 = conv_flops(8, 8, 4*C, 4*C)
    print('Scat A: {:.1f}Mflops'.format((layer1 + layer2 + conv2_1 + conv2_2
                                         + conv3_1 + conv3_2)/1e6))

    # ScatNet B
    C = 96
    layer1 = scatmix_flops(32, 32, 3)
    layer2 = scatmix_flops(16, 16, 21)
    conv2_1 = conv_flops(8, 8, 147, 2*C)
    conv2_2 = conv_flops(8, 8, 2*C, 2*C)
    conv3_1 = conv_flops(8, 8, 2*C, 4*C)
    conv3_2 = conv_flops(8, 8, 4*C, 4*C)
    print('Scat B: {:.1f}Mflops'.format((layer1 + layer2 + conv2_1 + conv2_2
                                         + conv3_1 + conv3_2)/1e6))

    # ScatNet C
    C = 96
    layer0 = conv_flops(32, 32, C, 16)
    layer1 = scat_flops(32, 32, 16)
    layer2 = scat_flops(16, 16, 16*7)
    conv2_1 = conv_flops(8, 8, 16*49, 2*C)
    conv2_2 = conv_flops(8, 8, 2*C, 2*C)
    conv3_1 = conv_flops(8, 8, 2*C, 4*C)
    conv3_2 = conv_flops(8, 8, 4*C, 4*C)
    print('Scat C: {:.1f}Mflops'.format((layer0 + layer1 + layer2 + conv2_1 +
                                         conv2_2 + conv3_1 + conv3_2)/1e6))

    # ScatNet D
    C = 96
    layer0 = conv_flops(32, 32, C, 16)
    layer1 = scatmix_flops(32, 32, 16)
    layer2 = scatmix_flops(16, 16, 16*7)
    conv2_1 = conv_flops(8, 8, 16*49, 2*C)
    conv2_2 = conv_flops(8, 8, 2*C, 2*C)
    conv3_1 = conv_flops(8, 8, 2*C, 4*C)
    conv3_2 = conv_flops(8, 8, 4*C, 4*C)
    print('Scat D: {:.1f}Mflops'.format((layer0 + layer1 + layer2 + conv2_1 +
                                         conv2_2 + conv3_1 + conv3_2)/1e6))

    # VGG16
    C = 64
    layers = [
        conv_flops(32, 32, 3, C),
        conv_flops(32, 32, C, C),
        conv_flops(16, 16, C, 2*C),
        conv_flops(16, 16, 2*C, 2*C),
        conv_flops(8, 8, 2*C, 4*C),
        conv_flops(8, 8, 4*C, 4*C),
        conv_flops(8, 8, 4*C, 4*C),
        conv_flops(4, 4, 4*C, 8*C),
        conv_flops(4, 4, 8*C, 8*C),
        conv_flops(4, 4, 8*C, 8*C),
        conv_flops(2, 2, 8*C, 8*C),
        conv_flops(2, 2, 8*C, 8*C),
        conv_flops(2, 2, 8*C, 8*C),
    ]
    print('VGG16: {:.1f}Mflops'.format(sum(layers)/1e6))

    # Resnet 110:
    layer1 = conv_flops(32, 32, 3, 16)
    scale1 = 6 * 18 * conv_flops(32, 32, 16, 16)
    scale2 = 6 * 18 * conv_flops(16, 16, 32, 32)
    scale3 = 6 * 18 * conv_flops(8, 8, 64, 64)

    print('Resnet-110: {:.1f}Mflops'.format((layer1 +
                                             scale1+scale2+scale3)/1e6))

    # All Conv
    C = 96
    layer1 = conv_flops(32, 32, 3, C)
    layer2 = conv_flops(32, 32, C, C)
    sample1 = conv_flops(16, 16, C, C)
    layer3 = conv_flops(16, 16, C, 2*C)
    layer4 = conv_flops(16, 16, 2*C, 2*C)
    sample2 = conv_flops(8, 8, 2*C, 2*C)
    layer5 = conv_flops(8, 8, 2*C, 2*C)
    layer6 = conv_flops(8, 8, 2*C, 2*C, L=1)
    print('ALl Conv: {:.1f}Mflops'.format((layer1 + layer2 +
                                           sample1+layer3 +
                                           layer4+sample2+layer5+layer6)/1e6))


if __name__ == '__main__':
    main()
