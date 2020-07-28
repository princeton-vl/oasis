import numpy as np
import cv2

def compress_png(data_float32, out_pngname):
    '''
        Use png encoding to reduce the size of the saved data

        data_float32: single precision 32 bit 2D/3D numpy array
        out_pngname: png filename
    '''
    if len(data_float32.shape) == 3:
        height, width, channel = data_float32.shape
    else:
        height, width = data_float32.shape
        channel = 1


    data_byte = data_float32.tobytes()
    data_uint16 = np.frombuffer(data_byte, dtype=np.uint16)
    data_uint16 = data_uint16.reshape((height, width * 2, channel))

    cv2.imwrite(out_pngname, data_uint16, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    

def decompress_png(in_filename):

    # img = Image.open(in_filename)
    # width, height = img.size
    data_uint16 = cv2.imread(in_filename, -1)
    height, width = data_uint16.shape[:2]
    if len(data_uint16.shape) > 2:
        channel = data_uint16.shape[2]
    else:
        channel = 1

    data_byte = data_uint16.tobytes()
    data_float = np.frombuffer(data_byte, dtype=np.float32)
    data_float = data_float.reshape((height, int(width / 2), channel))


    return np.squeeze(data_float)

def unit_test():
    out_data = np.random.random((1024, 768)).astype(np.float32)
    compress_png(out_data, "debug.png")
    in_data = decompress_png("debug.png")
    np.savez("debug.npz", out_data)
    np.save("debug.npy", out_data)
    in_data = np.squeeze(in_data)
    print (in_data.flatten() == out_data.flatten())



if __name__ == "__main__":
    unit_test()
