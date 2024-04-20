
import argparse
import time

from threading import Thread
import requests
import uvicorn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-p", "--port", type=int,default="-1")
    parser.add_argument("-ci", "--client_id", type=int,default="0")
    args = parser.parse_args()
    
    port = 8001 + args.client_id
    
    if args.port != -1:
        port = args.port

    def set_client_id():
        while True:
            try:
                requests.get(f"http://localhost:{port}/?client_id={args.client_id}")
            except Exception as error:
                # print("error", error)
                pass

            time.sleep(1)
    
    worker = Thread(target=set_client_id, args=(), daemon=True)
    worker.start()
        

    print(f"Start client {args.client_id} with PORT: {port}")
    
    uvicorn.run('client:app', port=port, reload=True)