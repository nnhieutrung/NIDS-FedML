
import argparse
import uvicorn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-p", "--port", type=int,default="8000")
    args = parser.parse_args()

    print(f"Start server with PORT: {args.port}")
    uvicorn.run('server:app', port=args.port, reload=True)
