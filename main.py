import argparse
import time

import numpy as np
import pyopencl as cl
import algosdk

from typing import NamedTuple

KEY_SIZE = 32


class Batch(NamedTuple):
    prefix: str
    count: int
    seeds: np.ndarray
    run_kernel_event: cl.event_info


def run(args: argparse.Namespace):
    benchmark: bool = args.benchmark
    prefix: str = args.prefix
    count: int = args.batch
    left: int = args.count

    ctx = cl.create_some_context()

    mf = cl.mem_flags

    with open(args.kernel, mode="r") as f:
        source = f.read()

    program = cl.Program(ctx, source).build(options="-cl-std=CL3.0")

    prefix_bytes = prefix.encode("utf-8")

    seed_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=count * KEY_SIZE)
    prefix_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.HOST_WRITE_ONLY, size=len(prefix_bytes))
    counts_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, size=count)

    kernel = program.ed25519_create_keypair
    queue = cl.CommandQueue(ctx)

    cl.enqueue_copy(queue, prefix_buffer, prefix_bytes)

    counts = np.empty(count, dtype='uint8')

    def send_next_batch() -> Batch:
        seeds = np.random.randint(0, 255, count * KEY_SIZE, dtype='uint8')
        cl.enqueue_copy(queue, seed_buffer, seeds, is_blocking=False)
        batch = Batch(
            prefix=prefix,
            count=count,
            seeds=seeds,
            run_kernel_event=kernel(queue, [count], None, seed_buffer, prefix_buffer, counts_buffer)
        )

        return batch

    batch = send_next_batch()

    start = None
    if benchmark:
        start = time.time()

    total = 0
    found = 0

    while found < left:
        copy_counts = cl.enqueue_copy(queue, counts, counts_buffer, is_blocking=False)

        next_batch = send_next_batch()

        cl.wait_for_events(
            [batch.run_kernel_event, copy_counts])

        total += batch.count

        seeds = batch.seeds
        expected = len(batch.prefix)
        indices = np.where(counts == expected)[0]

        for i in indices:
            phrases = algosdk.mnemonic._from_key(bytes(seeds[i * KEY_SIZE:(i + 1) * KEY_SIZE]))
            pk = algosdk.mnemonic.to_private_key(phrases)
            pk_addr = algosdk.account.address_from_private_key(pk)

            print("%s,%s" % (pk_addr, phrases))
            found += 1

            if found == left:
                break

        batch = next_batch

    if benchmark:
        end = time.time()
        delta = end - start

        avg = 0
        if delta > 0:
            avg = total / delta

        print("Total: %d, matching: %d, time: %s, avg: %d keys/s" % (total, found, delta, avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", required=True, type=str)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch", type=int, default=1024 * 64)
    parser.add_argument("--kernel", type=str, default="kernel.cl")
    parser.add_argument("--count", type=int, default=1)

    args = parser.parse_args()

    run(args)
