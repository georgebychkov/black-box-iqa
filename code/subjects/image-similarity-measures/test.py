#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from image_similarity_measures.quality_metrics import metric_functions
import numpy
import csv, sys

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_path", type=str)
    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    bps = 3
    if args.width * args.height <= 0:
       raise RuntimeException("unsupported resolution")

    all_metrics = sorted(metric_functions.items())
    writer = csv.DictWriter(sys.stdout, fieldnames=metric_functions.keys())
    writer.writeheader()

    with open(args.ref_path, 'rb') as ref_bgr24, open(args.dist_path, 'rb') as dist_bgr24:
        while True:
            ref = ref_bgr24.read(args.width * args.height * bps)
            dist = dist_bgr24.read(args.width * args.height * bps)
            if len(ref) == 0 and len(dist) == 0:
                break
            if len(ref) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream ref_path")
            if len(dist) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream dist_path")

            ref = numpy.frombuffer(ref, dtype='uint8').reshape((args.height,args.width,bps))
            dist = numpy.frombuffer(dist, dtype='uint8').reshape((args.height,args.width,bps))

            writer.writerow( dict( (metric, float(metric_func(ref, dist))) for metric, metric_func in all_metrics ) )

if __name__ == "__main__":
   main()
