package com.company;

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
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

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
        return str.substring(n) + str.substring(0,n);
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
    public String ReverseSentence(String str) {

    }


}
