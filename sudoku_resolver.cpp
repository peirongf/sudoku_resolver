#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stack>
#include <iostream>
#include <algorithm>

const int board_slice_y_start = 320;
const int board_slice_y_end   = 1760;
const int digits_slice_y_start= 2125;
const int digits_slice_y_end  = 2270;
const int digits_slice_x_start= 15;
const int digits_slice_x_end  = 1425;

using namespace cv;
using namespace std;

void print_usage(const char * cmd)
{
    printf("Usage: %s filename\n", cmd);
}

char * parse_options(int argc, char * argv[])
{
    if (argc <= 0) return NULL;
    else return argv[0];
}

void parse_sudoku_board(const char * filename, Mat &board)
{
    Mat gray;
    Mat digits[9];
    Mat board_area;
    Mat digits_area;
    size_t board_pos;
    size_t digits_num;

    gray = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!gray.data) {
        printf("Invalid filename\n");
        return;
    }
    gray(Rect(0, board_slice_y_start,
            gray.size().width, board_slice_y_end - board_slice_y_start)
        ).copyTo(board_area);
    threshold(board_area, board_area, 210, 255, THRESH_BINARY);
    gray(Rect(digits_slice_x_start, digits_slice_y_start,
             digits_slice_x_end - digits_slice_x_start,
             digits_slice_y_end - digits_slice_y_start)
        ).copyTo(digits_area);
    threshold(digits_area, digits_area, 120, 255, THRESH_BINARY_INV);
    rectangle(digits_area, Point(0, 0),
             Point(digits_area.cols - 1, digits_area.rows - 1), Scalar(0), 5);

    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;

    digits_num = 9;
    findContours(digits_area.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t k = 0; k < contours.size(); k ++) {
        vector<vector<Point> > contours0;
        if (contourArea(Mat(contours[k])) < 10000) {
            continue;
        }
        digits_area(boundingRect(Mat(contours[k]))).copyTo(digits[digits_num - 1]);
        subtract(Scalar::all(255), digits[digits_num - 1], digits[digits_num - 1]);
        findContours(digits[digits_num - 1].clone(), contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        digits[digits_num - 1](boundingRect(Mat(contours0[0]))).copyTo(digits[digits_num - 1]);
        subtract(Scalar::all(255), digits[digits_num - 1], digits[digits_num - 1]);
        resize(digits[digits_num - 1], digits[digits_num - 1], Size(20, 40));
        digits_num --;
    }

    board = Mat::zeros(9, 9, CV_8UC1);
    findContours(board_area.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t k = 0; k < contours.size(); k ++) {
        Mat mat;
        vector<vector<Point> > contours0;
        if (contourArea(Mat(contours[k])) < 10000) {
            continue;
        }
        board_area(boundingRect(Mat(contours[k]))).copyTo(mat);
        subtract(Scalar::all(255), mat, mat);
        findContours(mat.clone(), contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours0.size() > 0) {
            double ncc_max = 0;
            size_t matched_digit = 0;
            mat(boundingRect(Mat(contours0[0]))).copyTo(mat);
            subtract(Scalar::all(255), mat, mat);
            resize(mat, mat, Size(20, 40));
            for (size_t i = 0; i < 9; i ++) {
                Mat result;
                double minVal, maxVal;
                Point minLoc, maxLoc;
                matchTemplate(mat, digits[i], result, CV_TM_CCORR_NORMED);
                minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
                if (ncc_max < minVal) {
                    ncc_max = minVal;
                    matched_digit = i + 1;
                }
            }
            board.data[80 - k] = matched_digit;
        }
    }
}

void all_row(Mat board, size_t k, vector<size_t> &v)
{
    size_t row = k / 9;

    v.clear();
    for (size_t i = 0; i < 9; i ++) {
        if (board.data[row * 9 + i] != 0 && i != k % 9) {
            v.push_back(board.data[row * 9 + i]);
        }
    }
}

void all_column(Mat board, size_t k, vector<size_t> &v)
{
    size_t col = k % 9;

    v.clear();
    for (size_t i = 0; i < 9; i ++) {
        if (board.data[i * 9 + col] != 0 && i != k / 9) {
            v.push_back(board.data[i * 9 + col]);
        }
    }
}

void all_grid(Mat board, size_t k, vector<size_t> &v)
{
    size_t x, y, grid_x, grid_y, ingrid_x, ingrid_y, row, col;

    x = k % 9;
    y = k / 9;
    grid_x = x / 3;
    grid_y = y / 3;
    v.clear();
    for (size_t i = 0; i < 9; i ++) {
        ingrid_x = i % 3;
        ingrid_y = i / 3;
        row = grid_y * 3 + ingrid_y;
        col = grid_x * 3 + ingrid_x;
        if (board.data[row * 9 + col] != 0 && (row * 9 + col) != k) {
            v.push_back(board.data[row * 9 + col]);
        }
    }
}

void build_solution_set(Mat &board, vector<size_t> solution[])
{
    vector<size_t> row_set;
    vector<size_t> col_set;
    vector<size_t> grid_set;

    for (size_t k = 0; k < 81; k ++) {
        if (board.data[k] != 0) {
            solution[k].push_back(board.data[k]);
        } else {
            all_row(board, k, row_set);
            all_column(board, k, col_set);
            all_grid(board, k, grid_set);
            for (size_t i = 1; i <= 9; i ++) {
                if (std::find(row_set.begin(), row_set.end(), i) == row_set.end() &&
                    std::find(col_set.begin(), col_set.end(), i) == col_set.end() &&
                    std::find(grid_set.begin(), grid_set.end(), i) == grid_set.end()) {
                    solution[k].push_back(i);
                }
            }
            if (solution[k].size() == 1) {
                board.data[k] = solution[k][0];
            }
        }
    }
}

bool check_conflicts(Mat board, size_t k, size_t candidate)
{
    vector<size_t> row_set;
    vector<size_t> col_set;
    vector<size_t> grid_set;

    all_row(board, k, row_set);
    all_column(board, k, col_set);
    all_grid(board, k, grid_set);
    if (std::find(row_set.begin(), row_set.end(), candidate) != row_set.end() ||
        std::find(col_set.begin(), col_set.end(), candidate) != col_set.end() ||
        std::find(grid_set.begin(), grid_set.end(), candidate) != grid_set.end()) {
        return true;
    }
    return false;
}

size_t find_next_candidate(Mat board, size_t index, vector<size_t> possible_values)
{
    vector<size_t>::iterator start = possible_values.begin();
    if (board.data[index] != 0) {
        start = std::find(possible_values.begin(), possible_values.end(), board.data[index]);
        start ++;
    }
    for (vector<size_t>::iterator it = start; it != possible_values.end(); it ++) {
        if (!check_conflicts(board, index, *it)) {
            return *it;
        }
    }
    return 0;
}

void pop_find_candidate(Mat board, stack<size_t> &running_grid, vector<size_t> possible_matrix[], size_t &index, size_t &candidate)
{
    index = 0;
    candidate = 0;
    while (!running_grid.empty()) {
        index = running_grid.top();
        running_grid.pop();
        candidate = find_next_candidate(board, index, possible_matrix[index]);
        if (candidate > 0) {
            break;
        } else {
            board.data[index] = 0;
        }
    }
}

void resolve_enforce_search(Mat &board)
{
    size_t index;
    vector<size_t> possible_matrix[81];
    stack<size_t> running_grid;

    build_solution_set(board, possible_matrix);

    index = 0;
    while (index < 81) {
        if (possible_matrix[index].size() > 1) {
            size_t candidate = find_next_candidate(board, index, possible_matrix[index]);
            if (candidate == 0) {
                board.data[index] = 0;
                pop_find_candidate(board, running_grid, possible_matrix, index, candidate);
            }
            board.data[index] = candidate;
            running_grid.push(index);
        }
        index ++;
    }
}

void resolve_sudoku(const char * filename)
{
    Mat board;
    parse_sudoku_board(filename, board);
    if (!board.data) {
        printf("failed to load the sudoku problem from file: %s\n", filename);
        return;
    } else {
        resolve_enforce_search(board);
        cout << board << endl;
    }
}

int main(int argc, char * argv[])
{
    char * filename = NULL;
    filename = parse_options(argc - 1, argv + 1);
    if (filename == NULL) {
        print_usage(argv[0]);
        return -1;
    } else {
        resolve_sudoku(filename);
    }
    return 0;
}
