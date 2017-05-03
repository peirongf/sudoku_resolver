import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Sudoku automatic resolver.')
    parser.add_argument('filename',metavar='filename', type=str, help='filename of Sudoku problem snapshot picture')
    args = parser.parse_args()
    return args

def load_file(filename):
    gray_img = cv2.imread(filename, 0)
    #gray_img = cv2.GaussianBlur(gray_img, (7,7),0)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    ret, board = cv2.threshold(gray_img[320:1760, :].copy(), 210, 255, cv2.THRESH_BINARY)
    ret, digits= cv2.threshold(gray_img[2125:2270, 15:1425].copy(), 120, 255, cv2.THRESH_BINARY_INV)
    #board=cv2.dilate(board,kernel)
    #digits=cv2.dilate(digits,kernel)
    h,w=digits.shape[:2]
    cv2.rectangle(digits,(0,0), (w-1, h-1), 0, 5)
    return board, digits

def mse(image1, image2):
    err = np.sum((image1.astype('float') - image2.astype('float')) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err

def parse_board(board, digits):
    results = np.zeros(81, dtype = np.uint8)
    zero_counts = 0
    im2,contours,hierarchy=cv2.findContours(board.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 10000:
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        roi = board[y:y+h,x:x+w]
        im2,con,hierarchy = cv2.findContours(255-roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(con) > 0:
            x, y, w, h = cv2.boundingRect(con[0])
            board_digit = 255-cv2.resize(roi[y:y+h, x:x+w].copy(), (20, 40))
            min_err = sys.maxint
            matched_digit = 0
            for j in range(len(digits)):
                err = mse(board_digit, digits[j])
                if min_err > err:
                    min_err = err
                    matched_digit = j + 1
            results[80 - i] = matched_digit
        else:
            zero_counts += 1
    return results, zero_counts

def parse_digits(digits):
    results = []
    im2,contours,hierarchy=cv2.findContours(digits.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 10000:
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        digit_roi = digits[y:y+h, x:x+w]
        im2,con,hierarchy = cv2.findContours(255-digit_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(con[0])
        results.insert(0,255-cv2.resize(digit_roi[y:y+h, x:x+w].copy(), (20, 40)))
    return results

def next_zero(board, index):
    found = False
    list = np.arange(81)
    if index in np.arange(80):
        index += 1
    else:
        index = 0
    search_list = list[index:]
    if index > 0:
        search_list = np.append(search_list, list[0:list])
    for i in search_list:
        if board[i] == 0:
            index = i
            found = True
            break
    return found, index

def all_row(board, index):
    results = []
    row = index / 9
    for i in np.arange(9):
        if board[row * 9 + i] != 0 and i != index % 9:
            results.append(board[row * 9 + i])
    return results

def all_column(board, index):
    results = []
    col = index % 9
    for i in np.arange(9):
        if board[i * 9 + col] != 0 and i != index / 9:
            results.append(board[i * 9 + col])
    return results

def all_grid(board, index):
    x = index % 9
    y = index / 9
    grid_y = y / 3
    grid_x = x / 3
    results = []
    for i in np.arange(9):
        ingrid_y = i / 3
        ingrid_x = i % 3
        row = grid_y * 3 + ingrid_y
        col = grid_x * 3 + ingrid_x
        if board[row * 9 + col] != 0 and index != (row * 9 + col):
            results.append(board[row * 9 + col])
    return results

def reduce_more(board, index, grid_solution, possible_digits):
    cur_x = index % 9
    cur_y = index / 9
    cur_grid_x = cur_x / 3
    cur_grid_y = cur_y / 3
    cur_ingrid_x = cur_x - cur_grid_x * 3
    cur_ingrid_y = cur_y - cur_grid_y * 3
    possible_results = [set(possible_digits)] * 9
    for x in np.arange(9):
        if x / 3 == cur_grid_x:
            continue
        for ingrid_y in np.arange(3):
            for ingrid_x in np.arange(3):
                row = ingrid_y + cur_grid_y * 3
                col = ingrid_x + cur_grid_x * 3
                num = board[row * 9 + x]
                ingrid_num = board[row * 9 + col]
                if num > 0 and ingrid_num == 0 and num in possible_digits and num in possible_results[ingrid_y * 3 + ingrid_x]:
                    grid_solution[num - 1] -= 1
                    possible_results[ingrid_y * 3 + ingrid_x] = possible_results[ingrid_y * 3 + ingrid_x] - {num}
    for y in np.arange(9):
        if y / 3 == cur_grid_y:
            continue
        for ingrid_x in np.arange(3):
            for ingrid_y in np.arange(3):
                #if ingrid_y != cur_ingrid_y and ingrid_x != cur_ingrid_x:
                row = ingrid_y + cur_grid_y * 3
                col = ingrid_x + cur_grid_x * 3
                num = board[y * 9 + col]
                ingrid_num = board[row * 9 + col]
                if num > 0 and ingrid_num == 0 and num in possible_digits and num in possible_results[ingrid_y * 3 + ingrid_x]:
                    grid_solution[num - 1] -= 1
                    possible_results[ingrid_y * 3 + ingrid_x] = possible_results[ingrid_y * 3 + ingrid_x] - {num}

def resolve_by_inference(board):
    by_row = False
    no_update_count = 0
    while board.sum() < 45 * 9:
        by_row = not by_row
        no_update_count += 1
        for y in np.arange(9):
            for x in np.arange(9):
                if by_row:
                    row = y
                    col = x
                else:
                    row = x
                    col = y
                index = row * 9 + col
                if board[index] > 0:
                    continue
                row_set = all_row(board, index)
                col_set = all_column(board, index)
                grid_set= all_grid(board, index)
                all_digits = set.union(set(row_set), set(col_set), set(grid_set))
                possible_digits = set(np.arange(9) + 1) - all_digits
                if len(possible_digits) == 1:
                    board[index] = next(iter(possible_digits))
                    no_update_count = 0
                else:
                    grid_solution = np.ones(9) * (9 - len(grid_set))
                    for i in np.arange(9):
                        if (i + 1) in all_digits:
                            grid_solution[i] = 0
                    reduce_more(board, index, grid_solution, possible_digits)
                    for i in np.arange(9):
                        if grid_solution[i] == 1 and (i + 1) not in all_digits:
                            board[index] = i + 1
                            no_update_count = 0
                        elif grid_solution[i] == 1 and (i + 1) in all_digits:
                            print 'something is wrong ...'
                            sys.exit(1)
        if no_update_count > 0:
            break;

def generate_solution_matrix(board):
    solution = [list()] * 81
    for index in np.arange(81):
        if board[index] != 0:
            solution[index] = list([board[index]])
        else:
            row_set = all_row(board, index)
            col_set = all_column(board, index)
            grid_set= all_grid(board, index)
            solution[index] = list(set(np.arange(9) + 1) - set(row_set) - set(col_set) - set(grid_set))
        #solution[index].append(index)
    return solution

def check_conflicts(board, index, candidate):
    row_set = all_row(board, index)
    col_set = all_column(board, index)
    grid_set= all_grid(board, index)
    return candidate in set.union(set(row_set), set(col_set), set(grid_set))

def find_next_candidate(board, index, possible_values):
    if board[index] == 0:
        start = 0
    else:
        start = possible_values.index(board[index])
        start += 1
    for i in possible_values[start:]:
        if not check_conflicts(board, index, i):
            return i
    return 0

def pop_find_candidate(board, stack, possible_matrix):
    index = 0
    candidate = 0
    while len(stack) > 0:
        index = stack.pop()
        candidate = find_next_candidate(board, index, possible_matrix[index])
        if candidate > 0:
            break
        else:
            board[index] = 0
    return index, candidate

def resolve_by_enforce_search(board):
    possible_matrix = generate_solution_matrix(board)
    index = 0
    stack = []
    while index < 81:
        if len(possible_matrix[index]) > 1:
            candidate = find_next_candidate(board, index, possible_matrix[index])
            if candidate == 0:
                board[index] = 0
                index, candidate = pop_find_candidate(board, stack, possible_matrix)
            board[index] = candidate
            stack.append(index)
        index += 1

def sudoku_resolver(board, zeros):
    if board.sum() < 45 * 9:
        resolve_by_inference(board)
        if board.sum() < 45 * 9:
            resolve_by_enforce_search(board)

def resolve_sudoku(filename):
    board,digits = load_file(filename)
    digits_results = parse_digits(digits)
    board_results, zeros = parse_board(board, digits_results)
    sudoku_resolver(board_results, zeros)
    board_results.shape = [9, 9]
    print "Resolved: \n", board_results

if __name__ == "__main__":
    args = parse_args()
    resolve_sudoku(args.filename)
