package com.company;

import com.sun.source.tree.Tree;

import java.lang.reflect.Array;
import java.util.*;

public class Solution {

    // 二维数组中的查找
    /*
     * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从
     * 左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请
     * 完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     */
    public boolean Find(int target, int[][] array) {
        int a = 0;
        int b = array[0].length - 1;
        int num;
        while (a > array.length && b > -1) {
            num = array[a][b];
            if (num < target) {
                a++;
            } else if (num == target) {
                return true;
            } else {
                b--;
            }
        }
        return false;
    }

    // 替换空格
    /* 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy
     */
    public String replaceSpace(StringBuffer str) {
//        return str.toString().replaceAll(" ", "%20");
        List<String> temp = new ArrayList<>();
        String tem = "";
        int num = 0;
        while (num < str.length()) {
            if (str.charAt(num) != ' ') {
                tem += str.charAt(num);
            } else {
                temp.add(tem);
                tem = "";
            }
            num++;
        }
        temp.add(tem);
        String result = temp.get(0);
        for (int i = 1; i < temp.size(); i++) {
            result += "%20" + temp.get(i);
        }
        return result;
    }


    // 从尾到头打印链表
    /* 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。*/
    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> result = new ArrayList<>();
        while (listNode != null) {
            result.add(0, listNode.val);
            listNode = listNode.next;
        }
        return result;
    }

    // 重建二叉树
    /* 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，
     * 则重建二叉树并返回
     * 前序遍历：根、左、右
     * 中序遍历：左、根、右
     * 后序遍历：左、右、根
     */
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode next;

        TreeNode(int x) {
            val = x;
        }
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        int gen = pre[0];
        TreeNode result = new TreeNode(gen);
        int num = 0;
        while (num < in.length) {
            if (in[num] == gen) {
                result.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, num + 1), Arrays.copyOf(in, num));
                result.right = reConstructBinaryTree(Arrays.copyOfRange(pre, num + 1, pre.length), Arrays.copyOfRange(in, num + 1, in.length));
                break;
            }
            num++;
        }
        return result;
    }


    // 用两个栈实现队列
    /* 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*/
//    Stack<Integer> stack1 = new Stack<Integer>();
//    Stack<Integer> stack2 = new Stack<Integer>();
//
//    public void push(int node) {
//        while (!stack2.isEmpty()) {
//            stack1.push(stack2.pop());
//        }
//        stack1.push(node);
//        while (!stack1.isEmpty()) {
//            stack2.push(stack1.pop());
//        }
//    }
//
//    public int pop() {
//        return stack2.pop();
//    }

    // 旋转数组的最小数字
    /* 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0
     */
    public int minNumberInRotateArray(int[] array) {
        int num = array.length - 1;
        int start = 0;
        int result = array[start];
        if (result < array[num])
            return result;
        int middle = num / 2;
        if (result > array[middle]) {
            return minNumberInRotateArray(Arrays.copyOfRange(array, start, middle));
        }
        while (middle < num) {
            middle++;
            result = Math.min(result, array[middle]);
        }
        return result;
    }

    // 斐波那契数列：0，1，1，2，3，5，8，13，。。。。==> dp(n) = dp(n - 1) + dp(n - 2)
    /* 大家都知道斐波那契数列，现在要求输入一个整数n，
     * 请你输出斐波那契数列的第n项（从0开始，第0项为0）。
     * n<=39
     */
    public int Fibonacci(int n) {
        //O(2^n) O(1)
        /*
        if(n < 2)
            return n;
        return Fibonacci(n-1)+Fibonacci(n-2);
        */
        //O(n) O(n)
        /*
        int[] result = new int[40];
        result[0] = 0;
        result[1] = 1;
        for (int i = 2; i < 40; i++) {
            result[i] = result[i - 1] + result[i - 2];
        }
        return result[n];
        */
        int a = 0;
        int b = 1;
        while (n-- > 0) {
            b += a;
            a = b - a;
        }
        return a;
    }

    // 跳台阶
    /* 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。*/
    // 与斐波那契数列相似，不同的是第0 项为1
    public int JumpFloor(int target) {
        int a = 1;
        int b = 1;
        while (target-- > 0) {
            b += a;
            a = b - a;
        }
        return a;
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int temp = target - nums[i];
            if (map.containsKey(temp))
                return new int[]{i, map.get(temp)};
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }


    /* 给你一个字符串 text，你需要使用 text 中的字母来拼凑尽可能多的单词 "balloon"（气球）。
     * 字符串 text 中的每个字母最多只能被使用一次。请你返回最多可以拼凑出多少个单词 "balloon"。
     */
    public int maxNumberOfBalloons(String text) {
        if (text.length() < 7) {
            return 0;
        }
        int[] check = {0, 0, 0, 0, 0};//b,a,l,o,n
        for (int i = 0; i < text.length(); i++) {
            switch (text.charAt(i)) {
                case 'b':
                    check[0]++;
                    break;
                case 'a':
                    check[1]++;
                    break;
                case 'l':
                    check[2]++;
                    break;
                case 'o':
                    check[3]++;
                    break;
                case 'n':
                    check[4]++;
                    break;
            }
        }
        check[2] /= 2;
        check[3] /= 2;
        int count = check[0];
        for (int i = 1; i < 5; i++) {
            count = Math.min(count, check[i]);
        }
        return count;
    }

    // 变态跳台阶
    /* 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法*/
    // f(n) = f(n - 1) + f(n - 2) + f(n - 3) + ... + f(3) + f(2) + f(1) = 2 ^ (n - 1)
    // f(n) = 2f(n - 1)
    public int JumpFloorII(int target) {
//        return (int) Math.pow(2,target-1);
        int result = 1;
        while (--target > 0) {
            result *= 2;
        }
        return result;
    }

    // 矩形覆盖
    /* 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
     * 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     */
    // 同跳台阶，只是0的时候应该返回0，数列为：0，1，2，3，5，8，13，。。。。。
    public int RectCover(int target) {
        if (target < 4)
            return target;
        int a = 1;
        int b = 1;
        while (target-- > 0) {
            b += a;
            a = b - a;
        }
        return a;
    }

    // 二进制中1的个数
    /* 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。*/
    /* 解答：在机器中，整数都是以二进制补码的形式存在的，运算也是，n & (n - 1)是个小技巧，实际上相当把最右边的1 置换成0
     * 比如 7 = 00000 0111,循环为：（8位为例，实际上计算机中是32位）
     * result = 1,n = 0000 0111 & 0000 0110 = 0000 0110;
     * result = 2,n = 0000 0110 & 0000 0101 = 0000 0100;
     * result = 3,n = 0000 0100 & 0000 0011 = 0000 0000;
     * 比如 -7 = 1111 1001，循环为：（8位为例，实际上计算机中是32位）
     * result = 1,n = 1111 1001 & 1111 1000 = 1111 1000;
     * result = 2,n = 1111 1000 & 1111 0111 = 1111 0000;
     * result = 3,n = 1111 0000 & 1110 1111 = 1110 0000;
     * result = 4,n = 1110 0000 & 1101 1111 = 1100 0000;
     * result = 5,n = 1100 0000 & 1011 1111 = 1000 0000;
     * result = 6,n = 1000 0000 & 0111 1111 = 0000 0000;
     */
    public int NumberOf1(int n) {
        int result = 0;
        n = n & 0xffffffff;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }

    // 数值的整数次方
    /* 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * 保证base和exponent不同时为0
     */
    public double Power(double base, int exponent) {
        if (base == 0) {
            return 0;
        }
        if (exponent == 0) {
            return 1;
        }
        double a = 1, b = base;
        int num = exponent;
        exponent = Math.abs(exponent);
        while (exponent > 1) {
            if (exponent % 2 == 1) {
                a *= b;
                exponent--;
            } else {
                b *= b;
                exponent /= 2;
            }
        }
        if (num > 0) {
            return a * b;
        }
        return 1 / (a * b);
    }

    // 调整数组顺序使奇数位于偶数前面
    /* 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
     * 使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后
     * 半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     */
    /* 方法一：把奇偶数分别放两个List，最后放回array中
     * 方法二：扫描一遍，用i，j两个指针指向同一个值，从array[0]开始找，
     * 每当i找到偶数时，j开始往后找i以后的第一个奇数，然后i到j这一段后移一位，
     * 奇数放到i处，i++，当j++到num = array.length时，直接break无需再后移
     */
    public void reOrderArray(int[] array) {
//        int num = array.length;
//        List<Integer> oid = new ArrayList<>();
//        List<Integer> notOid = new ArrayList<>();
//        int i = 0;
//        while (i < num){
//            if (array[i] % 2 == 0){
//                notOid.add(array[i]);
//            }else {
//                oid.add(array[i]);
//            }
//            i++;
//        }
//        i = 0;
//        int j = 0;
//        for (;i < oid.size();i++)
//        {
//            array[i] = oid.get(i);
//        }
//        for (;j < notOid.size();j++){
//            array[i + j] = notOid.get(j);
//        }
        int num = array.length;
        int i = 0, j = 0, temp;
        while (i < num && j < num) {
            if (array[i] % 2 == 0) {
                j = i;
                while (j < num) {
                    if (array[j] % 2 == 0) {
                        j++;
                    } else {
                        break;
                    }
                }
                if (j == num) {
                    return;
                }
                temp = array[j];
                while (i < j) {
                    array[j] = array[j - 1];
                    j--;
                }
                array[i] = temp;
            }
            i++;
        }
    }

    // 链表中倒数第k个结点
    /* 输入一个链表，输出该链表中倒数第k个结点。*/
    /*
     * 方法一：遍历查出链表长度，返回count - k
     * 方法二：用快慢指针————i，j，j先走k - 1，然后i和j一起走，j走到末尾时，返回j
     */
    public ListNode FindKthToTail(ListNode head, int k) {
//        ListNode findk = head;
//        int count = 0;
//        while (null != findk){
//            findk = findk.next;
//            count ++;
//        }
//        if(count < k)
//            return null;
//        for (int i = 0; i < count - k; i++) {
//            head = head.next;
//        }
//        return head;
        ListNode i = head;
        ListNode j = head;
        int count = 0;
        while (null != j) {
            j = j.next;
            count++;
            if (count > k)
                i = i.next;
        }
        if (k > count)
            return null;
        return i;
    }

    // 反转链表
    /* 输入一个链表，反转链表后，输出新链表的表头*/
    // pre指针指向head的前一个结点，last指针指向head的next结点，不断移位即可
    public ListNode ReverseList(ListNode head) {
        if (null != head) {
            ListNode pre = null;
            ListNode last = head.next;
            while (null != last) {
                head.next = pre;
                pre = head;
                head = last;
                last = last.next;
            }
            head.next = pre;
        }
        return head;
    }

    // 合并两个排序的链表
    /* 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。*/
    /* 方法一：递归————两个表头结点对比，小的拿出来，list.next = Merge（list1.next，list2）直至为空就返回
     * 方法二：非递归————
     * */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (null == list1 && null != list2)
            return list2;
        if (null != list1 && null == list2)
            return list1;
        if (null == list1 && null == list2)
            return null;
        if (list1.val < list2.val) {
            list1.next = Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = Merge(list1, list2.next);
            return list2;
        }

//        ListNode result = new ListNode(-1);
//        ListNode temp = result;
//        while (null != list1 && null != list2) {
//            if (list1.val < list2.val) {
//                temp.next = list1;
//                list1 = list1.next;
//            } else {
//                temp.next = list2;
//                list2 = list2.next;
//            }
//            temp = temp.next;
//        }
//        if (null != list1) {
//            temp.next = list1;
//        }
//        if (null != list2)
//            temp.next = list2;
//        return result.next;
    }

    // 树的子结构
    /* 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）*/
    // 递归遍历判断，判断root1中是否有root2，另外写一个方法判断是否相等
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (null == root1 || null == root2)
            return false;
        return isequal(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    boolean isequal(TreeNode root1, TreeNode root2) {
        if (null == root2)
            return true;
        if (null == root1)
            return false;
        if (root1.val != root2.val)
            return false;
        return isequal(root1.left, root2.left) && isequal(root1.right, root2.right);
    }


    // 二叉树的镜像
    /* 操作给定的二叉树，将其变换为源二叉树的镜像。*/
    // 递归替换即可
    public void Mirror(TreeNode root) {
        if (null != root) {
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            Mirror(root.left);
            Mirror(root.right);
        }
    }

    // 顺时针打印矩阵
    /* 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
     * 例如，如果输入如下4 X 4矩阵：
     * 1   2   3   4
     * 5   6   7   8
     * 9   10  11  12
     * 13  14  15  16
     * 则依次打印出数字
     * 1,2,3,4,  8,12,16, 15,14,13, 9,5,6,7, 11,10.
     */
    // 就顺着打印思路写就行
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        if (null == matrix)
            return null;
        ArrayList<Integer> resultList = new ArrayList<>();
        int imin = 0;
        int imax = matrix.length - 1;
        int jmin = 0;
        int jmax = matrix[0].length - 1;
        int i, j;
        while (imin <= imax && jmin <= jmax) {
            for (j = jmin; j <= jmax; j++) {
                resultList.add(matrix[imin][j]);
            }
            for (i = imin + 1; i <= imax; i++) {
                resultList.add(matrix[i][jmax]);
            }
            if (imin < imax) {
                for (j = jmax - 1; j >= jmin; j--) {
                    resultList.add(matrix[imax][j]);
                }
            }
            if (jmin < jmax) {
                for (i = imax - 1; i >= imin + 1; i--) {
                    resultList.add(matrix[i][jmin]);
                }
            }
            imin++;
            imax--;
            jmin++;
            jmax--;
        }
        return resultList;
    }

    //包含min函数的栈
    /*定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。*/
    //每次都将最小元素min与当前栈同时push进栈，pop的时候同时pop两个
    Stack<Integer> stack = new Stack<>();
    int minmin;

    public void push(int node) {
        if (stack.empty()) {
            minmin = node;
        } else {
            minmin = stack.peek() > node ? node : stack.peek();
        }
        stack.push(node);
        stack.push(minmin);
    }

    public void pop() {
        stack.pop();
        stack.pop();
    }

    public int top() {
        int temp = stack.pop();
        int topNum = stack.peek();
        stack.push(temp);
        return topNum;
    }

    public int min() {
        return stack.peek();
    }

    //栈的压入、弹出序列
    /* 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个
     * 序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈
     * 序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的
     * 弹出序列。（注意：这两个序列的长度是相等的）
     */
    //模拟pushA压入栈，即遇到popA就与栈顶元素对比，相同就pop掉，不同就pushA再继续对比
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        Stack<Integer> temp = new Stack<>();
        int orderLength = pushA.length;
        int i = 0;
        int j = 0;
        while (i < orderLength && j < orderLength) {
            temp.push(pushA[i]);
            while (!temp.empty() && temp.peek() == popA[j]) {
                temp.pop();
                j++;
            }
            i++;
        }
        return temp.empty();
    }

    // 从上往下打印二叉树
    /* 从上往下打印出二叉树的每个节点，同层节点从左至右打印。*/
    // 利用队列，把root order进队列，再拿出来，把left和right order进队列
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        Queue<TreeNode> treeNodeQueue = new LinkedList<>();
        treeNodeQueue.offer(root);
        while (!treeNodeQueue.isEmpty()) {
            TreeNode temp = treeNodeQueue.poll();
            if (null != temp) {
                treeNodeQueue.offer(temp.left);
                treeNodeQueue.offer(temp.right);
                list.add(temp.val);
            }
        }
        return list;
    }

    // 二叉搜索树的后序遍历序列
    /* 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同
     */
    /* 后序遍历：左 ==> 右 ==> 根
     * 因为是二叉搜索树，所以左子树一定小于根，右子树一定大于根，
     * 后序遍历最后一个一定是根结点，以此为界限能分成大小两部分，
     * 递归继续分直至数目为1或者2时为true，否则为false
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        int length = sequence.length - 1;
        if (length == -1)
            return false;
        if (length == 0 || length == 1)
            return true;
        int check = sequence[length];
        int middle = length - 1;
        boolean ifsmall = false;
        for (int i = middle; i >= 0; i--) {
            if (!ifsmall) {
                if (sequence[i] > check) {
                    middle--;
                } else {
                    ifsmall = true;
                }
            } else if (sequence[i] > check) {
                return false;
            }
        }
        if (middle == length - 1 || middle == -1) {
            return VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, length));
        }
        return VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, middle + 1)) &&
                VerifySquenceOfBST(Arrays.copyOfRange(sequence, middle + 1, length - middle));
    }

    // 二叉树中和为某一值的路径
    /* 输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和
     * 为输入整数的所有路径。路径定义为从树的根结点开始往下一直到
     * 叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，
     * 数组长度大的数组靠前)
     */
    // 根据题目要求，一定要到叶子结点，否则就不算
    public ArrayList<ArrayList<Integer>> result = new ArrayList<>();
    public ArrayList<Integer> list = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return result;
        }
        list.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null) {
            result.add(new ArrayList<>(list));
        }
        FindPath(root.left, target);
        FindPath(root.right, target);
        list.remove(list.size() - 1);
        return result;
    }


    //复杂链表的复制
    /* 输入一个复杂链表（每个节点中有节点值，以及两个指针，
     * 一个指向下一个节点，另一个特殊指针指向任意一个节点），
     * 返回结果为复制后复杂链表的head。（注意，输出结果中请
     * 不要返回参数中的节点引用，否则判题程序会直接返回空）
     */
    // 方法一：用Map存已经访问过的结点，每次访问pHead的next时就在map里先找一遍，有就直接用map里面的，没有就重新new，
    // 加入map中（map的键值对关系为（被复制的结点：复制出来的新结点），本来想用List，后面发现List只能返回是否存在，不方便直接找到该结点）
    // 方法二：在旧链表中复制新链表，把新旧链表拆开成两个链表，返回新链表
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null)
            return null;
//        RandomListNode q = null;
//        Map<RandomListNode, RandomListNode> map = new HashMap<>();
//        RandomListNode newHead = new RandomListNode(pHead.label);       //复制的新链表头结点
//        q = newHead;
//        map.put(pHead, newHead);
//        while (pHead != null) {
//            if (pHead.next != null && map.containsKey(pHead.next))
//                q.next = map.get(pHead.next);
//            else {
//                if (pHead.next != null) {
//                    RandomListNode temp = new RandomListNode(pHead.next.label);
//                    map.put(pHead.next, temp);
//                    q.next = temp;
//                }
//            }
//            if (pHead.random != null && map.containsKey(pHead.random))
//                q.random = map.get(pHead.random);
//            else {
//                if (pHead.random != null) {
//                    RandomListNode temp = new RandomListNode(pHead.random.label);
//                    map.put(pHead.random, temp);
//                    q.random = temp;
//                }
//            }
//            pHead = pHead.next;
//            q = q.next;
//        }
//        return newHead;
        RandomListNode newHead = pHead;
        while (newHead != null) {
            RandomListNode newNode = new RandomListNode(newHead.label);
            newNode.next = newHead.next;
            newHead.next = newNode;
            newHead = newNode.next;
        }
        newHead = pHead;
        while (newHead != null) {
            newHead.next.random = newHead.random == null ? null : newHead.random.next;
            newHead = newHead.next.next;
        }
        RandomListNode oldHead = pHead;
        newHead = oldHead.next;
        while (oldHead != null) {
            RandomListNode temp = oldHead.next;
            oldHead.next = temp.next;
            temp.next = temp.next == null ? null : temp.next.next;
            oldHead = oldHead.next;
        }
        return newHead;
    }

    //二叉搜索树与双向链表
    /*输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向*/
    // 中序遍历存结点，再转换左右指针
    List<TreeNode> Treelist = new ArrayList<>();

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null)
            return null;
        orderInMiddle(pRootOfTree);
        for (int i = 0; i < Treelist.size() - 1; i++) {
            Treelist.get(i).right = Treelist.get(i + 1);
            Treelist.get(i + 1).left = Treelist.get(i);
        }
        return Treelist.get(0);
    }

    public void orderInMiddle(TreeNode root) {
        if (root.left != null)
            orderInMiddle(root.left);
        Treelist.add(root);
        if (root.right != null)
            orderInMiddle(root.right);
    }

    //字符串的排列
    /* 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
     * 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所
     * 有字符串abc,acb,bac,bca,cab和cba。
     * 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
     */
    // 遍历字符串，固定第一个元素，第一个元素可以取a,b,c...全部取到，然后递归求解 ————f(a,b,c) = a f(b,c) + b f(a,c) + c f(a,b)  排序
    public ArrayList<String> Permutation(String str) {
        return PermutationHelp(new StringBuilder(str));
    }

    public ArrayList<String> PermutationHelp(StringBuilder str) {
        ArrayList<String> result = new ArrayList<>();
        List<Character> list = new ArrayList<>();
        if (str.length() == 1)
            result.add(str.toString());
        else {
            for (int i = 0; i < str.length(); i++) {
                char temp = str.charAt(i);
                if (!list.contains(temp)) {
                    list.add(temp);
                    str.deleteCharAt(i);
                    ArrayList<String> newResult = PermutationHelp(str);
                    for (int j = 0; j < newResult.size(); j++) {
                        result.add(temp + newResult.get(j));
                    }
                    str.insert(i, temp);
                }
            }
        }
        Collections.sort(result);
        return result;
    }

    // 数组中出现次数超过一半的数字
    /* 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数
     * 组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
     */
    /* 方法一：借助map计算每个数的个数，遍历map，返回
     * 方法二：超过数组长度一半，也就是说，该数比其余所有数的个数和大，遍历找出这个数，再遍历数它的个数
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        int num = array.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < num; i++) {
            if (map.containsKey(array[i])) {
                int temp = map.get(array[i]);
                temp++;
                map.replace(array[i], temp);
            } else {
                map.put(array[i], 1);
            }
        }
        int i = 0;
        for (Integer key : map.keySet()) {
            if (map.get(key) > num / 2)
                return key;
        }
        return 0;
//        int temp = array[0];
//        int count = 1;
//        for (int i = 1; i < array.length; i++) {
//            if (temp == array[i])
//                count++;
//            else {
//                count--;
//                if (count == 0) {
//                    temp = array[i];
//                    count = 1;
//                }
//            }
//        }
//        count = 0;
//        for (int i = 0; i < num; i++) {
//            if (array[i] == temp)
//                count++;
//        }
//        if (count > num / 2)
//            return temp;
//        return 0;
    }

    // 最小的K个数
    /* 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。*/
    // 先对前K个数进行排序，再将后面的数一一与第k个数比较，大的就下一个，小就把第k个数踢掉，插到顺序的位置，最后返回前k个数
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> list = new ArrayList<>();
        if (k > 0 && k < input.length + 1) {
            for (int i = 1; i < k; i++) {
                int j = i - 1;
                int unFindElement = input[i];
                while (j >= 0 && input[j] > unFindElement) {
                    input[j + 1] = input[j];
                    j--;
                }
                input[j + 1] = unFindElement;
            }
            for (int i = k; i < input.length; i++) {
                if (input[i] < input[k - 1]) {
                    int newK = input[i];
                    int j = k - 1;
                    while (j >= 0 && input[j] > newK) {
                        input[j + 1] = input[j];
                        j--;
                    }
                    input[j + 1] = newK;
                }
            }
            for (int i = 0; i < k; i++) {
                list.add(input[i]);
            }
        }
        return list;
    }

    //连续子数组的最大和
    /* HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。
     * 今天测试组开完会后,他又发话了:
     * 在古老的一维模式识别中,常常需要计算连续子向量的最大和,
     * 当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,
     * 是否应该包含某个负数,并期望旁边的正数会弥补它呢？
     * 例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
     * 给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
     */
    // 遍历一遍数组，记录每次的Max值和当前总和，如果当前总和小于0则直接等于下一个数组值（基础动态规划）
    public int FindGreatestSumOfSubArray(int[] array) {
        int max = array[0];
        int sum = 0;
        for (int i = 0; i < array.length; i++) {
            if (sum < 0)
                sum = array[i];
            else {
                sum += array[i];
            }
            max = Math.max(max, sum);
        }
        return max;
    }

    // 整数中1出现的次数（从1到n整数中1出现的次数）
    /* 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
     * 为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,
     * 但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,
     * 可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）
     */
    /* 个人感觉是道数学题：https://www.jianshu.com/p/1a43f78d185f (找了很多篇，这篇最最最为值得看） */
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        int base = 1;                           // 当前 位数 进位
        int round = n;                          // 高位
        while (round > 0) {                     // 如果 高位 大于0 循环
            int weight = round % 10;            // 当前 位数 值
            round = round / 10;                 // 高位 个数
            count += round * base;              // 当前 位数 1 出现次数 每一轮出现1的个数，个位1次，十位10次，百位出现100次
            if (weight == 1) {                  // 当前 位数 为1
                count += (n % base) + 1;
            } else if (weight > 1) {            // 当前 位数 大于1
                count += base;
            }
            base = base * 10;
        }
        return count;
//        int count=0;
//        for(int i=1;i<=n;i*=10)
//        {
//            //i表示当前分析的是哪一个数位
//            int a = n/i,b = n%i;
//            count += (a + 8) / 10 * i + ((a % 10 == 1) ? b + 1 : 0);
//        }
//        return count;
    }

    // 把数组排成最小的数
    /* 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，
     * 打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，
     * 则打印出这三个数字能排成的最小数字为321323。
     */
    /* 个人感觉是个排序问题；
     * 方法：用Collections自带的sort，重写比较函数；
     * 1️⃣直接用String自带的compareTo比较s1 + s2 与 s2 + s1
     * 2️⃣逐位数比较 s1 与 s2，短的循环比较长的，完全相等则返回0
     */
    public String PrintMinNumber(int[] numbers) {
        List<String> list = new ArrayList<>();
        for (int i : numbers) {
            list.add(String.valueOf(i));
        }
        Collections.sort(list, new Comparator<String>() {
            @Override
            public int compare(String s, String t1) {
//                return (s + t1).compareTo(t1 + s);
                int i = 0, j = 0;
                while (i < s.length() || j < t1.length()) {
                    if (j == t1.length())
                        j -= t1.length();
                    if (i == s.length())
                        i -= s.length();
                    if (s.charAt(i) < t1.charAt(j)) {
                        return -1;
                    } else if (s.charAt(i) > t1.charAt(j)) {
                        return 1;
                    }
                    i++;
                    j++;
                }
                return 0;
            }
        });
        StringBuilder result = new StringBuilder();
        for (String s : list) {
            result.append(s);
        }
        return result.toString();

    }

    // 丑数
    /* 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
     */
    // 方法一：生成（超时）
    // 方法二：拆分成 2x3y5z
    public int GetUglyNumber_Solution(int index) {
        if (index < 7)
            return index;
        int[] res = new int[index];
        res[0] = 1;
        int pos2 = 0;// 2的对列
        int pos3 = 0;// 3的对列
        int pos5 = 0;// 5的对列
        // 一个丑数*2/3/5还是丑数
        for (int i = 1; i < index; i++) {
            res[i] = Math.min(Math.min(res[pos2] * 2, res[pos3] * 3), res[pos5] * 5);
            if (res[i] == res[pos2] * 2) {
                pos2++;
            }
            if (res[i] == res[pos3] * 3) {
                pos3++;
            }
            if (res[i] == res[pos5] * 5) {
                pos5++;
            }
        }
        return res[index - 1];
    }

    /* 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.*/
    // 方法一：用Map存已经出现的Char和次数，遍历str返回第一个在map中value为1的i
    // 方法二：利用Char的ASCII码
    public int FirstNotRepeatingChar(String str) {
//        Map<Character,Integer> map = new HashMap<>();
//        char[] ch = str.toCharArray();
//        for (int i = 0; i < ch.length; i++) {
//            if(!map.containsKey(ch[i])){
//                map.put(ch[i],1);
//            }else {
//                int temp = map.get(ch[i]);
//                temp ++;
//                map.replace(ch[i],temp);
//            }
//        }
//        if (!map.isEmpty()){
//            for (int i = 0; i < ch.length; i++) {
//                if (map.get(ch[i]) == 1)
//                    return i;
//            }
//        }
//        return -1;
        if (str != null) {
            int[] count = new int[256];
            for (int i = 0; i < str.length(); i++)
                count[str.charAt(i)]++;
            for (int i = 0; i < str.length(); i++)
                if (count[str.charAt(i)] == 1)
                    return i;
        }
        return -1;
    }

    /* 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
     * 输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
     * 对于%50的数据,size<=10^4
     * 对于%75的数据,size<=10^5
     * 对于%100的数据,size<=2*10^5
     */
    // 方法一：暴力遍历，时间复杂度为O(n ^ 2) 超时
    // 方法二：归并排序，时间复杂度为O(log n)
    private int count;

    public int InversePairs(int[] array) {
        mergeSort(array, 0, array.length - 1);
        return count;
    }

    public void merge(int[] array, int low, int mid, int high) {
        int[] temp = new int[high - low + 1];
        int i = low;
        int j = mid + 1;
        int k = 0;
        // 把较小的数先移到新数组中
        while (i <= mid && j <= high) {
            if (array[i] <= array[j]) {
                temp[k++] = array[i++];
            } else {
                temp[k++] = array[j++];
                count = (count + mid - i + 1) % 1000000007;
            }
        }
        while (i <= mid) {
            temp[k++] = array[i++];
        }
        while (j <= high) {
            temp[k++] = array[j++];
        }
        for (int l = 0; l < k; l++) {
            array[low + l] = temp[l];
        }
    }

    public void mergeSort(int[] array, int low, int high) {
        if (low >= high)
            return;
        int mid = (low + high) / 2;
        mergeSort(array, low, mid);             // 左边
        mergeSort(array, mid + 1, high);   // 右边
        merge(array, low, mid, high);           // 左右归并
    }

    // 两个链表的第一个公共结点
    /* 输入两个链表，找出它们的第一个公共结点。*/
    // 循环遍历：pHead1----1，2，3，6，7  pHead2----4，5，6，7   ==> 公共结点 6，7 ，也就是说，变相的将两个链表变成长度相等，同时遍历递进即可
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null)
            return null;
        ListNode p = pHead1;
        ListNode q = pHead2;
        while (p != q) {
            p = p.next;
            q = q.next;
            if (p == null) p = pHead2;
            if (q == null) q = pHead1;
        }
        return p;
    }

    // 数字在排序数组中出现的次数
    /* 统计一个数字在排序数组中出现的次数*/
    // 方法一：直接遍历：两头同时遍历，相遇则停止
    // 方法二：先找最后一个k，再找第一个k，返回 j - i
    // 找k的时候可以用二分法找，时间复杂度为 O(log n)
    public int GetNumberOfK(int[] array, int k) {
//        int count = 0;
//        int i = 0, j = array.length - 1;
//        System.out.println(Arrays.binarySearch(array,k));
//        if(j % 2 == 0){
//            if(array[(j + 1)/2] == k)
//                count ++;
//        }
//        while (i < j) {
//            if (array[i] == k)
//                count++;
//            if (array[j] == k)
//                count++;
//            i++;
//            j--;
//        }
//        return count;
        int i = 0, j = array.length - 1;
        while (j >= i) {
            if (array[j] == k)
                break;
            j--;
        }
        if (j == -1)
            return 0;
        while (i < j) {
            if (array[i] == k)
                break;
            i++;
        }
        return j - i + 1;
    }

    // 二叉树的深度
    /* 输入一棵二叉树，求该树的深度。
     * 从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度*/
    // 递归即可
    public int TreeDepth(TreeNode root) {
        int depth = 0;
        if (root != null) {
            depth++;
            int left = TreeDepth(root.left);
            int right = TreeDepth(root.right);
            depth += Math.max(left, right);
            //方法二添加
            if (Math.abs(left - right) > 1) {
                isBalanced = false;
            } else
                isBalanced = true;
        }
        return depth;
    }

    // 平衡二叉树(左右子树高度差不超过1)
    /* 输入一棵二叉树，判断该二叉树是否是平衡二叉树。*/
    // 方法一：直接遍历每个节点
    // 方法二：后序遍历(在找子树高度的同时先判断该子树是否为平衡二叉树)
    boolean isBalanced = false;

    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null)
            return true;
//        if( Math.abs(TreeDepth(root.left) - TreeDepth(root.right)) <= 1 ) {
//            return (IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right));
//        } else {
//            return false;
//        }
        TreeDepth(root);
        return isBalanced;
    }

    // 数组中只出现一次的数字
    /* 一个整型数组里除了两个数字之外，其他的数字都出现了两次。
     * 请写程序找出这两个只出现一次的数字
     * num1,num2分别为长度为1的数组。传出参数
     * 将num1[0],num2[0]设置为返回结果
     */
    // 方法一：利用Map把重复的删除，最后留下的两个即答案
    // 方法二：遍历异或找出两个不同数的异或，移位与 (&) 找出最小的那一位1，以此为界分为两个子数组各自异或即可
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
//        List<Integer> numList = new ArrayList<>();
//        Map<Integer,Integer> map = new HashMap<>();
//        for (int i = 0; i < array.length; i++) {
//            if(map.containsKey(array[i])){
//                map.remove(array[i],array[i]);
//            }else {
//                map.put(array[i],array[i]);
//            }
//        }
//        for(int item:map.values())
//            numList.add(item);
//        num1[0] = numList.get(0);
//        num2[0] = numList.get(1);
        int temp = 0;
        for (int i = 0; i < array.length; i++) {
            temp ^= array[i];
        }
        int findOne = 1;
        while ((findOne & temp) == 0) {
            findOne <<= 1;
        }
        int result1 = 0, result2 = 0;
        for (int i = 0; i < array.length; i++) {
            if ((findOne & array[i]) == 0)
                result1 ^= array[i];
            else
                result2 ^= array[i];
        }
        num1[0] = result1;
        num2[0] = result2;
    }

    // 和为S的连续正数序列
    /* 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序 */
    // 方法一：循环找，小于sum就加，大于sum就减掉最前面的，等于sum就清空list从原本list中的第二个开始重新循环
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        int i = 0, j = 1, num = 0;
        while (i < j && j < sum) {
            list.add(j);
            num += j;
            while (num > sum) {
                i++;
                list.remove(new Integer(i));
                num -= i;
            }
            if (num == sum) {
                result.add(new ArrayList<>(list));
                j = list.get(0);
                i = j;
                list.clear();
                num = 0;
            }
            j++;
        }
        return result;
    }

    /* 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的*/
    // 方法一：首尾并进夹逼寻找
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        int i = 0, j = array.length - 1;
        ArrayList<Integer> list = new ArrayList<>();
        while (i < j) {
            if (array[i] + array[j] < sum)
                i++;
            else if (array[i] + array[j] > sum)
                j--;
            else {
                list.add(array[i]);
                list.add(array[j]);
                break;
            }
        }
        return list;
    }

    /* 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
     * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
     * 例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
     * 是不是很简单？OK，搞定它！
     */
    // 方法一：借助StringBuilder 的append和deleteCharAt循环实现
    // 方法二：直接截取拼接
    public String LeftRotateString(String str, int n) {
        n = n % str.length();
        if (n == 0)
            return str;
        return str.substring(n) + str.substring(0, n);
//        StringBuilder result = new StringBuilder(str);
//        while (n > 0) {
//            result.append(result.charAt(0));
//            result.deleteCharAt(0);
//            n --;
//        }
//        result.append(result.substring(0,n));
//        return result.delete(0,n).toString();
    }

    /* 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
     * 同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
     * 例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
     * Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
     */
    // 方法一：用String自带的split函数以空格为界分开，逆序遍历添加
    // 方法二：逆序遇到" "就加进result中
    public String ReverseSentence(String str) {
        StringBuilder result = new StringBuilder();
//        String[] temp = str.split(" ");
//        int num = temp.length - 1;
//        if (num < 1)
//            return str;
//        result.append(temp[num]);
//        for (int i =  num - 1; i >= 0 ; i--) {
//            result.append(" " + temp[i]);
//        }
//        return result.toString();
        int i = str.length() - 1;
        if (i < 1)
            return str;
        int j = 0;
        while (i > 0) {
            if (str.charAt(i) != ' ') {
                j++;
            } else {
                result.append(" " + str.substring(i + 1, i + 1 + j));
                j = 0;
            }
            i--;
        }
        result.append(" " + str.substring(0, j + 1));
        return result.toString().trim();
    }

    //扑克牌顺子
    /* LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...
     * 他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！
     * “红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成
     * 任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),
     * “So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何，
     * 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
     */
    /* 方法：遍历，有对子直接false ，排序后除0外计算相邻两个数的差值和，(假设5个数中有x个0，数据两两之差的总和为y，满足y-x<=4-x, 即y<=4即可。)*/
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length < 5) {
            return false;
        }
        Arrays.sort(numbers);
        int sum = 0;
        for (int i = 0; i < numbers.length - 1; i++) {
            if (numbers[i] == 0) {
                continue;
            } else if (numbers[i] == numbers[i + 1])
                return false;
            else {
                sum += numbers[i + 1] - numbers[i];
            }
        }
        if (sum <= 4)
            return true;
        return false;
    }

    //孩子们的游戏(圆圈中最后剩下的数)
    /* 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
     * HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。
     * 然后,他随机指定一个数m,让编号为0的小朋友开始报数。
     * 每次喊到  m-1  的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
     * 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,
     * 并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？
     * (注：小朋友的编号是从0到n-1)
     */
    // 约瑟夫环，用ArrayList构建环，每次remove编号为k % list.size()的即可
    public int LastRemaining_Solution(int n, int m) {
        if (n <= 0)
            return -1;
        ArrayList<Integer> start = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            start.add(i);
        }
        int k = 0;
        while (start.size() != 1) {
            k += m - 1;
            k %= start.size();
            start.remove(k);
        }
        return start.get(0);
    }

    // 求1+2+3+...+n
    /* 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。*/
    // 根据题目要求，换句话说只能用 & | ^ (利用 && 的短路原则来代替判断)
    public int Sum_Solution(int n) {
        int sum = 0;
        boolean ans = false;
        int a = 0;
        ans = (n != 0) && (a == (sum = Sum_Solution(n - 1)));
        return sum + n;
    }

    // 不用加减乘除做加法
    /* 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号*/
    // 模拟二进制加法，
    public int Add(int num1, int num2) {
        int sum = 0, carry = 0;
        while (num2 != 0) {
            sum = num1 ^ num2;
            carry = (num1 & num2) << 1;
            num1 = sum;
            num2 = carry;
        }
        return sum;
    }

    // 把字符串转换成整数
    /* 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0*/
    // 0 到 9 的ASCII码范围是48 到 57
    public int StrToInt(String str) {
        if (str.length() == 0)
            return 0;
        int i = 0;
        boolean ifSign = false;
        switch (str.charAt(0)) {
            case '-':
                ifSign = true;
                i = 1;
                break;
            case '+':
                i = 1;
                break;
            case '0':
                return 0;
            default:
                i = 0;
        }
        int sum = 0;
        for (; i < str.length(); i++) {
            if (str.charAt(i) < 48 || str.charAt(i) > 57)
                return 0;
            sum = sum * 10 + (str.charAt(i) - 48);
        }
        if (ifSign)
            sum *= -1;
        if ((ifSign && sum > 0) || (!ifSign && sum < 0))
            return 0;
        return sum;
    }

    /* 在一个长度为n的数组里的所有数字都在0到n-1的范围内。
     * 数组中某些数字是重复的，但不知道有几个数字是重复的。
     * 也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
     * 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，
     * 那么对应的输出是第一个重复的数字2*/
    //    numbers:     数组
    //    length:      数组长度
    //    duplication: 返回任意重复的一个，赋值duplication[0]
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            if (list.contains(numbers[i])) {
                duplication[0] = numbers[i];
                return true;
            } else {
                list.add(numbers[i]);
            }
        }
        return false;
    }

    // 构建乘积数组
    /* 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。*/
    /* 分成上下两个对角矩阵，左下为D，右上为C，则有：
     * C[i] = A[i + 1] * A[i + 2] * ... * A[n - 1] = C[i + 1] * A[i + 1]
     * D[i] = A[0] * A[1] * ... * A[i - 1] = D[i - 1] * A[i - 1]
     * B[i] = C[i] * D[i]
     * 两个for即可
     */
    public int[] multiply(int[] A) {
        int num = A.length;
        int[] B = new int[num];
        B[0] = 1;
        for (int i = 1; i < num; i++) {
            B[i] = B[i - 1] * A[i - 1];
        }
        int temp = 1;
        for (int i = num - 2; i >= 0; i--) {
            temp *= A[i + 1];
            B[i] *= temp;
        }
        return B;
    }

    /* 请实现一个函数用来匹配包括'.'和'*'的正则表达式。
     * 模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
     */
    // 方法一:用String自带的matches（注意char[]转String应该用String.valueOf）
    public boolean match(char[] str, char[] pattern) {

//        String s = String.valueOf(str);
//        return s.matches(String.valueOf(pattern));
        return isMatch(String.valueOf(str), String.valueOf(pattern));
    }

    public boolean isMatch(String text, String pattern) {
        //如果都为空则匹配成功
        if (pattern.isEmpty())
            return text.isEmpty();
        //第一个是否匹配上
        boolean first_match = (!text.isEmpty() && (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.'));

        if (pattern.length() >= 2 && pattern.charAt(1) == '*') {
            //看有没有可能,剩下的pattern匹配上全部的text
            //看有没有可能,剩下的text匹配整个pattern
            //isMatch(text, pattern.substring(2)) 指当p第二个为*时，前面的字符不影响匹配所以可以忽略，所以将*以及*之前的一个字符删除后匹配之后的字符，这就是为什么用pattern.substring(2)
            //如果第一个已经匹配成功，并且第二个字符为*时，这是我们就要判断之后的需要匹配的字符串是否是多个前面的元素（*的功能），这就是first_match && isMatch(text.substring(1), pattern))的意义
            return (isMatch(text, pattern.substring(2)) ||
                    (first_match && isMatch(text.substring(1), pattern)));
        } else {
            //没有星星的情况:第一个字符相等,而且剩下的text,匹配上剩下的pattern，没有星星且第一个匹配成功，那么s和p同时向右移动一位看是否仍然能匹配成功
            return first_match && isMatch(text.substring(1), pattern.substring(1));
        }
    }

    // 表示数值的字符串
    /* 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     * 例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
     * 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
     */
    // 标志小数点和指数
    // +-号后面必定为数字 或 后面为.（-.123 = -0.123）
    // +-号只出现在第一位或eE的后一位
    // .后面必定为数字 或为最后一位（233. = 233.0）
    // e/E后面必定为数字或+-号
    public boolean isNumeric(char[] str) {
        boolean point = false, exp = false;
        for (int i = 0; i < str.length; i++) {
            if (str[i] == '+' || str[i] == '-') {
                if (i + 1 == str.length || !(str[i + 1] >= '0' && str[i + 1] <= '9' || str[i + 1] == '.')) {
                    return false;
                }
                if (!(i == 0 || str[i - 1] == 'e' || str[i - 1] == 'E')) {
                    return false;
                }
            } else if (str[i] == '.') {
                if (point || exp || !(i + 1 < str.length && str[i + 1] >= '0' && str[i + 1] <= '9')) {
                    return false;
                }
                point = true;
            } else if (str[i] == 'e' || str[i] == 'E') {
                if (exp || i + 1 == str.length || !(str[i] >= '0' && str[i + 1] <= '9' || str[i + 1] == '+' || str[i + 1] == '-')) {
                    return false;
                }
                exp = true;
            } else if (str[i] >= '0' && str[i] <= '9') {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    // 字符流中第一个不重复的字符
    /* 请实现一个函数用来找出字符流中第一个只出现一次的字符。
     * 例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
     * 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
     * 如果当前字符流没有存在出现一次的字符，返回#字符。
     */
    //Insert one char from stringstream
    String str = "";

    public void Insert(char ch) {
        str += ch;
    }

    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce() {
        int[] array = new int[256];
        for (int i = 0; i < str.length(); i++) {
            array[str.charAt(i)]++;
        }
        for (int i = 0; i < str.length(); i++) {
            if (array[str.charAt(i)] == 1)
                return str.charAt(i);
        }
        return '#';
    }

    // 链表中环的入口结点
    /* 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。*/
    // 方法一：用List找出重复的第一个节点
    // 方法二：用快慢指针：先找到环，再找环长度，然后用找第k结点的方法找到起点
    public ListNode EntryNodeOfLoop(ListNode pHead) {
//        List<ListNode> list = new ArrayList<>();
//        while (pHead != null){
//            if(!list.contains(pHead)){
//                list.add(pHead);
//                pHead = pHead.next;
//            }else {
//                return pHead;
//            }
//        }
//        return null;
        boolean isNull = true;
        ListNode p = pHead;
        ListNode q = pHead;
        while (q != null && q.next != null) {
            p = p.next;
            q = q.next.next;
            if (p == q) {
                isNull = false;
                break;
            }
        }
        if (isNull)
            return null;
        int count = 1;
        q = q.next;
        while (p != q) {
            q = q.next;
            count++;
        }
        p = q = pHead;
        while (count > 0) {
            q = q.next;
            count--;
        }
        while (p != q) {
            p = p.next;
            q = q.next;
        }
        return p;
    }

    // 删除链表中重复的结点
    /* 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5 */
    // 方法一： 借助List存重复的结点，再遍历找出头结点，遍历删除重复结点
    // 方法二： 借助重新建立的新头结点（保证 -1 不会是原链表中的数）
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null)
            return pHead;
//        List<Integer> list = new ArrayList<>();
//        int checkNum = pHead.val;
//        ListNode check = pHead.next;
//        while (check!= null){
//            if(check.val == checkNum){
//                list.add(check.val);
//            }else {
//                checkNum = check.val;
//            }
//            check = check.next;
//        }
//        while (pHead != null){
//            if (list.contains(pHead.val))
//                pHead = pHead.next;
//            else
//                break;
//        }
//        if (pHead == null)
//            return pHead;
//        ListNode pre = pHead;
//        check = pre.next;
//        while (check != null){
//            if (list.contains(check.val)){
//                check = check.next;
//                pre.next = check;
//            }else {
//                pre = check;
//                check = check.next;
//            }
//        }
//        return pHead;
        ListNode head = new ListNode(-1);
        ListNode current = head;
        while (pHead != null) {
            ListNode check = pHead.next;
            boolean findSame = false;
            while (check != null && pHead.val == check.val) {
                check = check.next;
                findSame = true;
            }
            if (!findSame) {
                current.next = pHead;
                current = current.next;
            }
            pHead = check;
        }
        current.next = null;
        return head.next;
    }

    // 二叉树的下一个结点
    /* 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
     * 注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
     */
    // 中序遍历：左，中，右
    /* 1️⃣如果右子树非空，直接返回右子树最左结点
     * 2️⃣右子树空，所给结点为左子树：
     *    （1）所给结点为父结点的左子结点，返回父节点
     *    （2）所给结点为父结点的右子结点，一直向上找到结点是左子结点的父节点，返回此结点的父结点
     */
    public TreeNode GetNext(TreeNode pNode) {
        if (pNode.right != null) {
            TreeNode pRight = pNode.right;
            while (pRight.left != null)
                pRight = pRight.left;
            return pRight;
        }
        if (pNode.next != null) {
            if (pNode == pNode.next.left)
                return pNode.next;
            TreeNode pNext = pNode.next;
            while (pNext.next != null && pNext.next.right == pNext)
                pNext = pNext.next;
            return pNext.next;
        }
        return null;
    }

    // 对称的二叉树
    /* 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的*/
    // 方法一： 递归判断,把需要判断的子树传入判断
    // 方法二： 迭代判断，借助队列 (ArrayDeque 不能add null，LinkedList 能)
    boolean isSymmetrical(TreeNode pRoot) {
//        return isMirror(pRoot,pRoot);
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(pRoot);
        queue.add(pRoot);
        while (!queue.isEmpty()) {
            TreeNode p = queue.poll();
            TreeNode q = queue.poll();
            if (p == null && q == null)
                continue;
            else if (p == null || q == null)
                return false;
            if (p.val != q.val)
                return false;

            queue.add(p.left);
            queue.add(q.right);
            queue.add(p.right);
            queue.add(q.left);
        }
        return true;
    }

    boolean isMirror(TreeNode pRoot1, TreeNode pRoot2) {
        if (pRoot1 == null && pRoot2 == null)
            return true;
        if (pRoot1 == null || pRoot2 == null)
            return false;
        return (pRoot1.val == pRoot2.val) && isMirror(pRoot1.left, pRoot2.right) && isMirror(pRoot1.right, pRoot2.left);
    }

    // 按之字形顺序打印二叉树
    /* 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。*/
    // 用两个栈分别存奇数层与偶数层的结点，空结点不存
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        Stack<TreeNode> stack0 = new Stack<>();
        Stack<TreeNode> stack1 = new Stack<>();
        if (pRoot != null) {
            stack0.push(pRoot);
            while (stack0.size() != 0 || stack1.size() != 0) {
                while (stack0.size() != 0) {
                    TreeNode temp = stack0.pop();
                    if (temp.left != null)
                        stack1.push(temp.left);
                    if (temp.right != null)
                        stack1.push(temp.right);
                    list.add(temp.val);
                }
                result.add(new ArrayList<>(list));
                list.clear();
                while (stack1.size() != 0) {
                    TreeNode temp = stack1.pop();
                    if (temp.right != null)
                        stack0.push(temp.right);
                    if (temp.left != null)
                        stack0.push(temp.left);
                    list.add(temp.val);
                }
                if (!list.isEmpty()) {
                    result.add(new ArrayList<>(list));
                    list.clear();
                }
            }
        }
        return result;
    }

    // 把二叉树打印成多行
    /* 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。*/
    // 与上题类似
    ArrayList<ArrayList<Integer>> Print2(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        Deque<TreeNode> queue = new ArrayDeque<>();
        if (pRoot != null) {
            queue.offer(pRoot);
            while (queue.size() != 0) {
                int count = queue.size();
                for (int i = 0; i < count; i++) {
                    TreeNode temp = queue.poll();
                    if (temp.left != null)
                        queue.add(temp.left);
                    if (temp.right != null)
                        queue.add(temp.right);
                    list.add(temp.val);
                }
                if (!list.isEmpty()) {
                    result.add(new ArrayList<>(list));
                    list.clear();
                }
            }
        }
        return result;
    }

    // 序列化二叉树
    /* 请实现两个函数，分别用来序列化和反序列化二叉树
     * 二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，
     * 从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序
     * 的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空
     * 节点（#），以 ！ 表示一个结点值的结束（value!）。
     * 二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。
     */
    // 前序遍历利用递归存到String中，还原时同样利用递归，重点是利用index来标记结点
    public int index = -1;

    String Serialize(TreeNode root) {
        StringBuilder result = new StringBuilder();
        if (root == null)
            result.append("#!");
        else {
            result.append(root.val + "!");
            result.append(Serialize(root.left));
            result.append(Serialize(root.right));
        }
        return result.toString();
    }

    TreeNode Deserialize(String str) {
        index++;
        if (index >= str.length())
            return null;
        String[] check = str.split("!");
        TreeNode pNode = null;
        if (!check[index].equals("#")) {
            pNode = new TreeNode(Integer.valueOf(check[index]));
            pNode.left = Deserialize(str);
            pNode.right = Deserialize(str);
        }
        return pNode;
    }

    // 二叉搜索树的第k个结点
    /* 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。*/
    // 先计算树的结点个数，与k做比较，递归查找
    // 中序遍历，直接返回第k个
    List<TreeNode> kthNodeList = new ArrayList<>();

    TreeNode KthNode(TreeNode pRoot, int k) {
//        if (k > nodeNumber(pRoot) || k <= 0)
//            return null;
//        int check = k - nodeNumber(pRoot.left);
//        if (check == 1)
//            return pRoot;
//        else if (check <= 0)
//            return KthNode(pRoot.left,k);
//        else
//            return KthNode(pRoot.right,check - 1);
        midOrder(pRoot);
        return kthNodeList.get(k - 1);
    }

    int nodeNumber(TreeNode pRoot) {
        int num = 0;
        if (pRoot == null)
            return num;
        return nodeNumber(pRoot.left) + nodeNumber(pRoot.right) + 1;
    }

    void midOrder(TreeNode pRoot) {
        if (pRoot.left != null)
            midOrder(pRoot.left);
        kthNodeList.add(pRoot);
        if (pRoot.right != null)
            midOrder(pRoot.right);
    }


    // 数据流中的中位数
    /* 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
     * 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，
     * 使用GetMedian()方法获取当前读取数据的中位数。
     */
    // 插入的时候就排序，中位数直接返回中间值，排序用二分查找位置然后插入
    List<Integer> midNumber = new ArrayList<>();

    public void Insert(Integer num) {
        int count = midNumber.size();
        int i = 0;
        int j = count;
        int mid;
        if (j != 0) {
            while (i <= j) {
                mid = (j + i) / 2;
                if (midNumber.get(mid) > num) {
                    j = mid - 1;
                } else if (midNumber.get(mid) == num)
                    break;
                else {
                    i = mid + 1;
                    if (i == count)
                        break;
                }
            }
        }
        midNumber.add(i, num);
    }

    public Double GetMedian() {
        int num = midNumber.size();
        if (midNumber.size() % 2 == 1)
            return midNumber.get(num / 2).doubleValue();
        return (midNumber.get((num / 2) - 1) + midNumber.get(num / 2)) / 2.0;
    }

    // 滑动窗口的最大值
    /* 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
     * 例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，
     * 他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个：
     * {[2,3,4],2,6,2,5,1}，
     * {2,[3,4,2],6,2,5,1}，
     * {2,3,[4,2,6],2,5,1}，
     * {2,3,4,[2,6,2],5,1}，
     * {2,3,4,2,[6,2,5],1}，
     * {2,3,4,2,6,[2,5,1]}。
     */
    // 利用双端队列，保证队头始终为size中的最大，移动时检查队头是否在size中，不在就剔除，从后往前将队列中比新元素小的数的下标剔除
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        int length = num.length;
        ArrayList<Integer> result = new ArrayList<>();
        ArrayDeque<Integer> deq = new ArrayDeque<Integer>();
        if (length * size != 0 && length >= size) {
            for (int i = 0; i < num.length; i++) {
                if (!deq.isEmpty() && deq.getFirst() == i - size)
                    deq.removeFirst();
                while (!deq.isEmpty() && num[i] > num[deq.getLast()])
                    deq.removeLast();
                deq.addLast(i);
                if (i >= size - 1)
                    result.add(num[deq.getFirst()]);
            }
        }
        return result;
    }

    // 矩阵中的路径
    /* 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
     * 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
     * 如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
     * 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，
     * 因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子
     */
    // 用回溯法遍历，用新的数组visited标记是否访问过，利用check与rows和cols的关系来递进上下左右格子

    boolean[] visited = null;

    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        visited = new boolean[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i] == str[0]) {
                if (subHasPath(matrix, rows, cols, str, i, 0))
                    return true;
            }
        }
        return false;
    }

    public boolean subHasPath(char[] matrix, int rows, int cols, char[] str, int check, int current) {
        if (matrix[check] != str[current] || visited[check] == true)
            return false;
        if (current == str.length - 1)
            return true;
        visited[check] = true;
        int row = check / cols;
        int col = check % cols;
        if (row > 0 && subHasPath(matrix, rows, cols, str, check - cols, current + 1))  //往上
            return true;
        if (row < rows - 1 && subHasPath(matrix, rows, cols, str, check + cols, current + 1))// 往下
            return true;
        if (col > 0 && subHasPath(matrix, rows, cols, str, check - 1, current + 1)) //往左
            return true;
        if (col < cols - 1 && subHasPath(matrix, rows, cols, str, check + 1, current + 1))  //往右
            return true;
        visited[check] = false; // 取消访问，返回上一层
        return false;
    }

    // 机器人的运动范围
    /* 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，
     * 每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），
     * 因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
     */
    // 与上题类似但有一点点区别，因为是从(0,0)出发，也就是从左下角出发，所以只需要考虑 向上 和 向右 即可 ，也就是说只需要考虑 x + 1 和 y + 1 的回溯
    boolean[][] visit = null;

    public int movingCount(int threshold, int rows, int cols) {
        visit = new boolean[rows][cols];
        return movingCountSum(threshold, rows, cols, 0, 0);

    }

    public int movingCountSum(int threshold, int rows, int cols, int x, int y) {
        if (x < rows && y < cols && !visit[x][y] && ifArrive(x, y, threshold)) {
            visit[x][y] = true;
            return 1 + movingCountSum(threshold, rows, cols, x + 1, y) + movingCountSum(threshold, rows, cols, x, y + 1);
        }
        return 0;

    }

    public boolean ifArrive(int x, int y, int k) {
        int sum = 0;
        while (x != 0) {
            sum += x % 10;
            x /= 10;
        }
        while (y != 0) {
            sum += y % 10;
            y /= 10;
        }
        return sum > k ? false : true;
    }

    // 剪绳子
    /* 给你一根长度为n的绳子，请把绳子剪成m段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。、
     * 输入一个数n，意义见题面。（2 <= n <= 60）
     */
    // 方法一： dp
    // 方法二：递归
    // 方法三：数学方法————拆成 2 / 3
    public int cutRope(int target) {
//        return cutRope(target,0);
//        if (target < 4)
//            return target - 1;
//        int[] result = new int[target + 1];
//        result[0] = 0;
//        result[1] = 1;
//        result[2] = 2;
//        result[3] = 3;
//
//        //自底向上开始求解
//        int max = 0;
//        for (int i = 4; i <= target; i++) {
//            max = 0;
//            for (int j = 1; j <= i / 2; j++) {
//                result[i] = Math.max(max,result[j] * result[i - j]);
//            }
//        }
//        max = result[target];
//        return max;
        if (target < 4)
            return target - 1;
        int threeTimes = target / 3;
        if (target - threeTimes * 3 == 1)
            threeTimes -= 1;
        int timeOfTwo = (target - threeTimes * 3) / 2;
        return fastPower(3,threeTimes) * fastPower(2,timeOfTwo);
    }
    public int fastPower(int a,int b){
        if (b == 0)
            return 1;
        int result = a;
        while (b != 1 && b != 0){
            if (b % 2 == 1){
                result *= a;
                b -= 1;
            }else {
                result *= result;
                a = result;
                b /= 2;
            }
        }
        return result;
    }

    public int cutRope(int target, int max) {
        for (int i = 1; i < target; i++) {
            max = Math.max(max, i * cutRope(target - i, target - i));
        }
        return max;
    }
}
