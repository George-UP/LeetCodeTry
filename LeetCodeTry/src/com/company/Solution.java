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
            if(!ifsmall){
                if (sequence[i] > check) {
                    middle --;
                }else {
                    ifsmall = true;
                }
            }else if (sequence[i] > check){
                return false;
            }
        }
        if (middle == length - 1 || middle == -1){
            return VerifySquenceOfBST(Arrays.copyOfRange(sequence,0,length));
        }
        return VerifySquenceOfBST(Arrays.copyOfRange(sequence,0,middle + 1)) &&
                VerifySquenceOfBST(Arrays.copyOfRange(sequence,middle + 1,length - middle));
    }

    // 二叉树中和为某一值的路径
    /* 输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和
     * 为输入整数的所有路径。路径定义为从树的根结点开始往下一直到
     * 叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，
     * 数组长度大的数组靠前)
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {

    }


}
