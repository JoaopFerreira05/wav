#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np

def read_pgm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header != 'P5':
            raise ValueError("Only binary PGM (P5) is supported.")
        
        line = f.readline().decode('ascii')
        while line.startswith('#'):
            line = f.readline().decode('ascii')
            
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('ascii').strip())
        
        if max_val == 255:
            dtype = np.uint8
        elif max_val == 65535:
            dtype = np.dtype('>u2')
        else:
            raise ValueError(f"Unsupported PGM max_val: {max_val}")
            
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((height, width))
        return data, max_val

def write_pgm(file_path, data, max_val):
    height, width = data.shape
    with open(file_path, 'wb') as f:
        f.write(f"P5\n{width} {height}\n{max_val}\n".encode('ascii'))
        if max_val == 255:
            f.write(data.astype(np.uint8).tobytes())
        else:
            f.write(data.astype(np.dtype('>u2')).tobytes())

def read_ppm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header != 'P6':
            raise ValueError("Only binary PPM (P6) is supported.")
            
        line = f.readline().decode('ascii')
        while line.startswith('#'):
            line = f.readline().decode('ascii')
            
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('ascii').strip())
        if max_val != 255:
            raise ValueError("Only 8-bit per channel PPM is currently supported.")
            
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape((height, width, 3))
        return data, max_val

def write_ppm(file_path, data, max_val):
    height, width, _ = data.shape
    with open(file_path, 'wb') as f:
        f.write(f"P6\n{width} {height}\n{max_val}\n".encode('ascii'))
        f.write(data.astype(np.uint8).tobytes())

def read_raw(file_path, width, height, bpp, endianness):
    if bpp == 8:
        dtype = np.uint8
    elif bpp == 16:
        dtype = np.dtype('>u2') if endianness == 'big' else np.dtype('<u2')
    else:
        raise ValueError("RAW bpp must be 8 or 16.")
        
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    
    if data.size != width * height:
        raise ValueError("File size does not match provided width and height dimensions.")
    return data.reshape((height, width))

def write_raw(file_path, data, bpp, endianness):
    if bpp == 8:
        out_data = data.astype(np.uint8)
    elif bpp == 16:
        out_data = data.astype(np.dtype('>u2') if endianness == 'big' else np.dtype('<u2'))
    else:
        out_data = data
    with open(file_path, 'wb') as f:
        f.write(out_data.tobytes())

def quantize(image, original_bpp, target_bits):
    if target_bits >= original_bpp:
        return image

    if target_bits <= 0:
        raise ValueError("Target bits must be greater than 0.")

    shift = original_bpp - target_bits

    quantized_levels = image >> shift

    return (quantized_levels << shift) + (1 << (shift - 1))

def main():
    parser = argparse.ArgumentParser(description="Uniform scalar quantization for RAW, PGM, or PPM images.")
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("output", help="Output image file path")
    parser.add_argument("format", choices=["raw", "pgm", "ppm"], help="Image format")
    parser.add_argument("bits", type=int, help="Target number of bits per pixel/channel after quantization")
    
    parser.add_argument("--width", type=int, help="Number of columns (required for RAW)")
    parser.add_argument("--height", type=int, help="Number of rows (required for RAW)")
    parser.add_argument("--bpp", type=int, choices=[8, 16], help="Bits per pixel (required for RAW)")
    parser.add_argument("--endian", choices=["big", "little"], default="big", help="Byte order for 16-bit RAW (default: big)")
    
    args = parser.parse_args()
    
    try:
        if args.format == "pgm":
            img, max_val = read_pgm(args.input)
            original_bpp = 8 if max_val == 255 else 16
            
            quant_img = quantize(img, original_bpp, args.bits)
            write_pgm(args.output, quant_img, max_val)
            print(f"PGM image successfully quantized to {args.bits} bits and saved to '{args.output}'.")
            
        elif args.format == "ppm":
            img, max_val = read_ppm(args.input)
            original_bpp = 8
            
            quant_img = quantize(img, original_bpp, args.bits)
            write_ppm(args.output, quant_img, max_val)
            print(f"PPM image successfully quantized to {args.bits} bits per channel and saved to '{args.output}'.")
            
        elif args.format == "raw":
            if not args.width or not args.height or not args.bpp:
                print("Error: RAW format requires --width, --height, and --bpp arguments.", file=sys.stderr)
                sys.exit(1)
                
            img = read_raw(args.input, args.width, args.height, args.bpp, args.endian)
            quant_img = quantize(img, args.bpp, args.bits)
            write_raw(args.output, quant_img, args.bpp, args.endian)
            print(f"RAW image successfully quantized to {args.bits} bits and saved to '{args.output}'.")
            
    except Exception as e:
        print(f"Error during quantization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()