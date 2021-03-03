import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// 303 (easy, T: 77.14%, S: 41.56%)
class NumArray {
    int[] sum_nums;
    public NumArray(int[] nums) {
        if (nums.length == 0) {
            return;
        }
        sum_nums = new int[nums.length];
        sum_nums[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            sum_nums[i] = sum_nums[i-1] + nums[i];
        }
    }

    public int sumRange(int i, int j) {
        int low = i == 0? 0 : sum_nums[i-1];
        int high = sum_nums[j];
        return high - low;
    }
}

// 304 (medium, T: 31.03%, S: 84.29%)
class NumMatrix {
    int[][] sum_matrix;

    public NumMatrix(int[][] matrix) {
        if (matrix.length == 0) {
            return;
        }
        sum_matrix = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                if (i == 0 && j == 0) {
                    sum_matrix[i][j] = matrix[i][j];
                } else if (i == 0) {
                    sum_matrix[i][j] = matrix[i][j] + sum_matrix[i][j-1];
                } else if (j == 0) {
                    sum_matrix[i][j] = matrix[i][j] + sum_matrix[i-1][j];
                } else {
                    sum_matrix[i][j] = matrix[i][j] + sum_matrix[i-1][j] + sum_matrix[i][j-1] - sum_matrix[i-1][j-1];
                }
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        if (row1 == 0 && col1 == 0) {
            return sum_matrix[row2][col2];
        } else if (row1 == 0) {
            return sum_matrix[row2][col2] - sum_matrix[row2][col1-1];
        } else if (col1 == 0) {
            return sum_matrix[row2][col2] - sum_matrix[row1-1][col2];
        } else {
            return sum_matrix[row2][col2] - sum_matrix[row1-1][col2] - sum_matrix[row2][col1-1] + sum_matrix[row1-1][col1-1];
        }
    }
}

public class Solution {

    // 29 (medium, T: 36.02%, S: 9.49%)
    public int divide(int dividend, int divisor) {
        if (dividend == 0) return 0;
        if (divisor == 1) return dividend;
        if (divisor == -1) {
            if (dividend == Integer.MIN_VALUE) {
                return Integer.MAX_VALUE;
            }
            return -dividend;
        }
        if (divisor == Integer.MIN_VALUE) {
            if (dividend == Integer.MIN_VALUE) {
                return 1;
            }
            return 0;
        }
        int count = 0;
        boolean is_negative = (divisor < 0 && dividend > 0) || (divisor > 0 && dividend < 0);
        if (dividend == Integer.MIN_VALUE) {
            count = 1;
            dividend = Integer.MAX_VALUE - Math.abs(divisor) + 1;
        }
        dividend = Math.abs(dividend);
        divisor = Math.abs(divisor);
        int[] counts = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};
        int[] notes = new int[32];
        notes[1] = divisor;
        int max_note = 1;

        while (dividend >= divisor) {
            if (dividend >= notes[max_note]) {
                dividend -= notes[max_note];
                count += counts[max_note];
                max_note += 1;
                notes[max_note] = notes[max_note-1] + notes[max_note-1];

            } else {
                for (int i = max_note-1; i > 0; --i) {
                    if (notes[i] < dividend) {
                        dividend -= notes[i];
                        max_note = i;
                        count += counts[max_note];
                    }
                }
            }
        }
        if (is_negative) {
            count = -count;
        }
        return count;
    }

    // 31 (medium, T: 98.26%, S: 94.14%)
    public void nextPermutation(int[] nums) {
        for (int i = nums.length - 2; i >= 0; --i) {
            if (nums[i] < nums[i+1]) {
                for (int j = nums.length-1; j > i; --j) {
                    if (nums[i] < nums[j]) {
                        int temp = nums[i];
                        nums[i] = nums[j];
                        nums[j] = temp;
                        break;
                    }
                }
                for (int j = 0; j < (nums.length-1-i)/2; ++j) {
                    int temp = nums[i+1+j];
                    nums[i+1+j] = nums[nums.length-1-j];
                    nums[nums.length-1-j] = temp;
                }
                return;
            }
        }
        for (int j = 0; j < nums.length/2; ++j) {
            int temp = nums[j];
            nums[j] = nums[nums.length-1-j];
            nums[nums.length-1-j] = temp;
        }
    }

    // 33 (medium, T: 10.86%, S: 5.32%)
    public int search(int[] nums, int target) {
        if (nums.length == 1) {
            return target == nums[0] ? 0 : -1;
        }
        int i = 0, j = nums.length-1;
        while (i < j) {
            int mid = (i+j+1) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            System.out.format("%d %d %d\n", i, j, mid);
            if (mid == j) {
                break;
            }
            if (nums[i] > nums[j]) {
                if (nums[i] < nums[mid]) {
                    if (nums[j] >= target) {
                        i = mid;
                    } else if (nums[mid] <= target) {
                        i = mid;
                    } else if (nums[mid] >= target) {
                        j = mid;
                    }
                } else {
                    if (nums[i] <= target) {
                        j = mid;
                    }
                    else if (nums[mid] >= target) {
                        j = mid;
                    }
                    else if (nums[mid] <= target) {
                        i = mid;
                    }
                }
            }
            else {
                if (nums[mid] > target) {
                    j = mid;
                }
                else if (nums[mid] < target) {
                    i = mid;
                }
                else {
                    return mid;
                }
            }
        }
        return nums[i] == target ? i : -1;
    }

    // 34 (medium, T: 100.00%, S: 31.14%)
    public int[] searchRange(int[] nums, int target) {
        if (nums.length ==  0) return new int[]{-1, -1};
        int li = 0, lj = nums.length-1;
        int left=-1, right=-1;
        while (li < lj) {
            left = (li+lj) / 2;
            if (nums[left] >= target) {
                lj = left;
            } else {
                li = left+1;
            }
        }
        int ri = 0, rj = nums.length-1;
        while (ri < rj) {
            right = (ri+rj+1) / 2;
            if (nums[right] <= target) {
                ri = right;
            } else {
                rj = right-1;
            }
        }
        if (nums[li] != target) {
            return new int[]{-1, -1};
        }
        return new int[]{li, rj};
    }

    // 36 (medium, T: 95.36%, S: 41.84%)
    public boolean isValidSudoku(char[][] board) {
        int[][] row = new int[9][10];
        int[][] col = new int[9][10];
        int[][] block = new int[9][10];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '0';
                    if (row[i][num] == num || col[j][num] == num || block[3*(i/3)+(j/3)][num] == num) {
//                        System.out.format("%d %d %d", i, j, num);
                        return false;
                    }
                    row[i][num] = num;
                    col[j][num] = num;
                    block[3*(i/3)+(j/3)][num] = num;
                }
            }
        }
        return true;
    }

    // 43 (medium, T: 43.31%, S: 39.38%)
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        StringBuilder[] nums = new StringBuilder[num1.length()];
        for (int i = 0; i < num1.length(); ++i) {
            nums[i] = new StringBuilder();
        }
        for (int i = num1.length()-1; i >= 0; --i) {
            int next = 0;
            for (int k = 0; k < num1.length()-1-i; k++) {
                nums[i].append(0);
            }
            for (int j = num2.length()-1; j >= 0; --j) {
                int result = (num1.charAt(i)-'0') * (num2.charAt(j)-'0') + next;
                nums[i].append(result % 10);
                next = result / 10;
            }
            if (next != 0) {
                nums[i].append(next);
            }
//            System.out.println(num1.charAt(i) + " " + num2 +" " + nums[i].toString());
        }
        StringBuilder res = new StringBuilder();
        int next = 0;
        for (int i = 0; i < nums[0].length(); ++i) {
            int result = next;
            for (int j = 0; j < nums.length; ++j) {
                if (nums[j].length() > i) {
                    result += nums[j].charAt(i) - '0';
                }
            }
            res.append(result % 10);
            next = result / 10;
        }
        res.reverse();
        if (next != 0) {
            res.insert(0, next);
        }
        return res.toString();
    }

    // 50 (medium, T: 98.82%, S: 36.02%)
    public double myPow(double x, long n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            return 1 / this.myPow(x, -n);
        }
        else {
            double mid = this.myPow(x, n / 2);

            return (n&1) == 1 ? mid*mid*x : mid*mid;
        }
    }

    // 53 (easy, T: 12.78%, S: 69.00%%)
    public int maxSubArray(int[] nums) {
        int[][] dp = new int[2][nums.length];
        dp[0][0] = nums[0];
        dp[1][0] = nums[0];
        for (int i = 1 ; i < nums.length; ++i) {
            dp[0][i] = Math.max(dp[0][i-1], dp[1][i-1]);
            dp[1][i] = Math.max(dp[1][i-1]+nums[i], nums[i]);
        }
        return Math.max(dp[0][nums.length-1], dp[1][nums.length-1]);
    }

    // 54 (medium, T: 100.0% S: 48.40%)
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        int row = matrix.length, col = matrix[0].length;
        int[][] direct = {
                {0, 1}, {1, 0}, {0, -1}, {-1, 0}
        };
        int i = 0, j = 0;
        int order_index = 0;
        boolean[][] visited_matrix = new boolean[row][col];
        for (int count = 0; count < row*col; count++) {
            result.add(matrix[i][j]);
            visited_matrix[i][j] = true;
            if (i + direct[order_index][0] >= row || j + direct[order_index][1] >= col ||
                    i + direct[order_index][0] < 0 || j + direct[order_index][1] < 0 ||
                    visited_matrix[i + direct[order_index][0]][j + direct[order_index][1]]) {
                order_index = (order_index + 1) % 4;
            }
            i += direct[order_index][0];
            j += direct[order_index][1];
        }
        return result;
    }

    // 70 (easy, T: 100.000%, S: 54.51%)
    public int climbStairs(int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return 1;
        }
        int a = 1, b = 1;
        for (int i = 1; i < n; ++i) {
            int temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    // 338 (medium, T: 99.95%, S: 80.05%)
    public int[] countBits(int num) {
        if (num == 0) return new int[]{0};
        int[] result = new int[num+1];
        result[0] = 0;;
        int max = 0, i = 1;
        while (true) {
            for (int j = 0; j <= max; ++j) {
                result[i+j] = result[j]+1;
                if (i+j == num) {
                    return result;
                }
            }
            i += max+1;
            max = i-1;
        }
    }

    // 896 (easy, T: 40.70%, S: 65.07%)
    public boolean isMonotonic(int[] A) {
        boolean gt = true, lt = true;
        for (int i = 1; i < A.length; ++i) {
            if (!gt && !lt) {
                return false;
            }
            if (A[i] > A[i-1]) {
                lt = false;
            }
            if (A[i] < A[i-1]) {
                gt = false;
            }
        }
        return gt || lt;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();

        System.out.println(solution.divide(-1010369383 ,-2147483648)); // 29
        int[] next_permutation = {1, 3, 2};
        solution.nextPermutation(next_permutation);
        System.out.println(Arrays.toString(next_permutation)); // 31
        System.out.println(solution.search(new int[]{3, 5, 1}, 1)); // 33
        System.out.println(Arrays.toString(solution.searchRange(new int[]{5}, 8))); // 34
        System.out.println(solution.multiply("987", "789")); // 43
        System.out.println(solution.myPow(2.0, -2)); // 50
        System.out.println(solution.maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4})); // 53
        System.out.println(solution.spiralOrder(new int[][]{{1, 2}, {3, 4}})); // 54
        System.out.println(solution.climbStairs(5)); // 70
    }
}

