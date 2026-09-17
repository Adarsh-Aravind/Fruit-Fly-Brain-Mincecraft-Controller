"""Let the fly brain play.

    python -m flybrain.run --pure-fly                 # connectome reflexes only
    python -m flybrain.run --checkpoint runs/default/latest.pt
"""
import argparse
import asyncio

import torch

from .bridge import Agent, BodyServer, Dashboard, control_loop


async def main_async(args):
    server = BodyServer(args.bodies, port=args.ws_port)
    await server.start()
    agent = Agent(args.bodies, window_ms=args.window_ms, shuffled=args.shuffled, pure_fly=args.pure_fly)
    if args.checkpoint and not args.pure_fly:
        ck = torch.load(args.checkpoint, map_location=agent.device, weights_only=False)
        agent.readout.load_state_dict(ck["readout"])
        print(f"loaded readout from {args.checkpoint} (update {ck.get('update')})")
    elif not args.pure_fly:
        print("no checkpoint given: the readout is untrained, so behaviour equals the pure fly")
    dash = Dashboard(agent, server, port=args.dashboard_port)
    await dash.start()
    await control_loop(agent, server, dash, hz=args.hz, greedy=args.greedy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=1)
    ap.add_argument("--checkpoint")
    ap.add_argument("--pure-fly", action="store_true")
    ap.add_argument("--shuffled", action="store_true", help="control: degree-preserving shuffled wiring")
    ap.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--dashboard-port", type=int, default=8000)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
