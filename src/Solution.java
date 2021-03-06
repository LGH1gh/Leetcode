import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

class MyQueue {

    Stack<Integer> in, out;
    /** Initialize your data structure here. */
    public MyQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        in.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if (out.empty()) {
            int size = in.size();
            for (int i = 0; i < size; ++i) {
                out.push(in.pop());
            }
        }
        return out.pop();

    }
    
    /** Get the front element. */
    public int peek() {
        if (out.empty()) {
            int size = in.size();
            for (int i = 0; i < size; ++i) {
                out.push(in.pop());
            }
        }

        return out.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return in.empty() && out.empty();
    }
}

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
    // 4 (hard, T: 82.41%, S: 72.23%)
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        Integer totalLength = nums1.length + nums2.length;
        Integer midIndex = totalLength / 2 + 1;
        Integer nums1Iter = -1, nums2Iter = -1;
        Integer mid1Num = 0, mid2Num = 0;

        for (int i = 0; i < midIndex; ++i) {
            if (nums1Iter + 1 == nums1.length) {
                nums2Iter++;
                mid1Num = mid2Num;
                mid2Num = nums2[nums2Iter];
            }
            else if (nums2Iter + 1 == nums2.length) {
                nums1Iter++;
                mid1Num = mid2Num;
                mid2Num = nums1[nums1Iter];
            }
            else {
                if (nums1[nums1Iter+1] < nums2[nums2Iter+1]) {
                    nums1Iter++;
                    mid1Num = mid2Num;
                    mid2Num = nums1[nums1Iter];
                }
                else {
                    nums2Iter++;
                    mid1Num = mid2Num;
                    mid2Num = nums2[nums2Iter];
                }
            }
        }

        if (totalLength % 2 == 0) {
            return 1.0 * (mid1Num + mid2Num) / 2.0;
        }
        else {
            return 1.0 * mid2Num;
        }
    }

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

    // 74 (medium, T: 100.00%, S: 68.67%)
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = 0;
        for (; row < matrix.length; ++row) {
            if (matrix[row][0] > target) {
                break;
            }
        }
        if (row == matrix.length) row = row - 1;
        else if (matrix[row][0] > target) {
            row = row - 1;
            if (row < 0) return false;
        }
        int col = 0;
        for (; col < matrix[0].length; ++col) {
            if (matrix[row][col] == target) {
                return true;
            }
            if (matrix[row][col] > target) {
                return false;
            }
        }
        return false;
    }

    // 81 (medium, T: 88.78%, S: 30.70%)
    public boolean searchII(int[] nums, int target) {
        return Arrays.stream(nums).boxed().collect(Collectors.toList()).contains(target);
    }

    // 90 (medium, T: 100.00%, S: 36.70%)
    public void subsetsWithDupHelper(int[] nums, int position, List<List<Integer>> result, List<Integer> temp) {
        if (position >= nums.length) {
            result.add(temp);
            return;
        }
        for (int i = position; i <= nums.length; ++i) {
            if (i == nums.length) {
                List<Integer> next = new ArrayList<>(temp);
                subsetsWithDupHelper(nums, i+1, result, next);
            }
            else if (i == position || nums[i] != nums[i-1]) {
                List<Integer> next = new ArrayList<>(temp);
                next.add(nums[i]);
                subsetsWithDupHelper(nums, i+1, result, next);
            }
            
        }
    }
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>());
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0 || nums[i] != nums[i-1]) {
                List<Integer> temp = new ArrayList<>();
                temp.add(nums[i]);
                subsetsWithDupHelper(nums, i+1, result, temp);
            }
        }
        return result;
    }

    // 190 (easy, T: 100.00%, S: 63.98%)
    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; ++i) {
            result = (result << 1) + (n & 1);
            n = n >>> 1;
        }
        return result;
    }

    // 213 (medium, T: 100.00%, S: 9.75%)
    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int[][] dp1 = new int[nums.length-1][2];
        dp1[0][0] = nums[0];
        dp1[0][1] = 0;
        for (int i = 1; i < nums.length-1; ++i) {
            dp1[i][0] = dp1[i-1][1] + nums[i];
            dp1[i][1] = Math.max(dp1[i-1][1], dp1[i-1][0]);
        }
        int[][] dp2 = new int[nums.length-1][2];
        dp2[0][0] = nums[1];
        dp2[0][1] = 0;
        for (int i = 2; i < nums.length; ++i) {
            dp2[i-1][0] = dp2[i-2][1] + nums[i];
            dp2[i-1][1] = Math.max(dp2[i-2][1], dp2[i-2][0]);
        }
        return Math.max(Math.max(dp1[nums.length-2][0], dp1[nums.length-2][1]), Math.max(dp2[nums.length-2][0], dp2[nums.length-2][1]));
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

    // 354 (hard, T: 19.11%, S: 5.12%)
    class MaxEnvelopes {
        MaxEnvelopes(int num, int[] last) {
            this.num = num;
            this.last = last;
        }
        int num;
        int[] last;
    }
    public int maxEnvelopes(int[][] envelopes) {
        Arrays.sort(envelopes, (int[] a, int[] b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            else return a[1] - b[1];
        });
        // System.out.println(Arrays.deepToString(envelopes));

        MaxEnvelopes[] dp = new MaxEnvelopes[envelopes.length];
        dp[0] = new MaxEnvelopes(1, envelopes[0]);
        int result = 1;
        for (int i = 1; i < envelopes.length; ++i) {
            int max = -1;
            for (int j = i-1; j >= 0; --j) {
                if (envelopes[j][0] < envelopes[i][0] && envelopes[j][1] < envelopes[i][1]) {
                    if (max == -1) max = j;
                    else if (dp[j].num > dp[max].num) max = j;
                }
            }
            if (max == -1) {
                dp[i] = new MaxEnvelopes(1, envelopes[i]);
            } else {
                dp[i] = new MaxEnvelopes(dp[max].num+1, envelopes[max]);
                if (result < dp[max].num+1) {
                    result = dp[max].num+1;
                }
                // System.out.printf("%d %d [%d %d] [%d %d] %d\n", max, i, envelopes[max][0], envelopes[max][1], envelopes[i][0], envelopes[i][1], dp[max].num+1);
            }
        }
        return result;
    }

    // 503 (medium, T: 72.39%, S: 7.57%)
    public int[] nextGreaterElements(int[] nums) {
        if (nums.length == 0) return new int[0];
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> mark = new Stack<>();
        int[] result = new int[nums.length];
        stack.push(nums[0]);
        mark.push(0);
        int max = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > max) {
                max = nums[i];
            }
            if (stack.peek() >= nums[i]) {
                stack.push(nums[i]);
                mark.push(i);
            }
            else {
                while (!stack.empty() && stack.peek() < nums[i]) {
                    stack.pop();
                    result[mark.pop()] = nums[i];
                }
                stack.push(nums[i]);
                mark.push(i);
            }
        }
        for (int i = 0; i < nums.length; ++i) {
            if (stack.empty()) {
                break;
            }
            if (stack.peek() == max) {
                stack.pop();
                result[mark.pop()] = -1;
            } else if (stack.peek() < nums[i]) {
                while (!stack.empty() && stack.peek() < nums[i]) {
                    stack.pop();
                    result[mark.pop()] = nums[i];
                }
            }
        }
        return result;
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

        // System.out.println(solution.divide(-1010369383 ,-2147483648)); // 29
        // int[] next_permutation = {1, 3, 2};
        // solution.nextPermutation(next_permutation);
        // System.out.println(Arrays.toString(next_permutation)); // 31
        // System.out.println(solution.search(new int[]{3, 5, 1}, 1)); // 33
        // System.out.println(Arrays.toString(solution.searchRange(new int[]{5}, 8))); // 34
        // System.out.println(solution.multiply("987", "789")); // 43
        // System.out.println(solution.myPow(2.0, -2)); // 50
        // System.out.println(solution.maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4})); // 53
        // System.out.println(solution.spiralOrder(new int[][]{{1, 2}, {3, 4}})); // 54
        // System.out.println(solution.climbStairs(5)); // 70
        System.out.println(solution.maxEnvelopes(new int[][]{{46,89},{50,53},{52,68},{72,45},{77,81}}
        )); // 354
    }
}
// {{2,100},{3,200},{4,300},{5,500},{5,400},{5,250},{6,370},{6,360},{7,380}}
// {{1,3},{3,5},{6,7},{6,8},{8,4},{9,5}}
// {{1,2},{2,3},{3,4},{3,5},{4,5},{5,5},{5,6},{6,7},{7,8}}
// {{46,89},{50,53},{52,68},{72,45},{77,81}}