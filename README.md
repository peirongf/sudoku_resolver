# sudoku_resolver
A sudoku resolver

Python version:
This is a python automatic sudoku resolver.
It takes a sudoku problem screenshot as an input and will resolve the sudoku automatically.
The image processing part will recognize the problem from the input image.
Right now it is not general for all images, but only accept portrait screenshot from this game:
https://play.google.com/store/apps/details?id=com.brainium.sudoku.free&hl=en

C++ version:
how to compile: g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` -o sudoku_resolver
how to run: ./sudoku_resolver filename
