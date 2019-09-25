package com.company;

import java.util.*;

public class Solution {

    //二维数组中的查找
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

    //替换空格
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


    从尾到头打印链表
    /*输入一个链表，按链表从尾到头的顺序返回一个ArrayList。*/
    public class ListNode {
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

    //重建二叉树
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


    //用两个栈实现队列
    /*用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*/
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        while (!stack2.isEmpty()) {
            stack1.push(stack2.pop());
        }
        stack1.push(node);
        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }
    }

    public int pop() {
        return stack2.pop();
    }

    //旋转数组的最小数字
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

    //斐波那契数列：0，1，1，2，3，5，8，13，。。。。==> dp(n) = dp(n - 1) + dp(n - 2)
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

    //跳台阶
    /*一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。*/
    //与斐波那契数列相似，不同的是第0 项为1
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

    //变态跳台阶
    /*一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法*/
    //f(n) = f(n - 1) + f(n - 2) + f(n - 3) + ... + f(3) + f(2) + f(1) = 2 ^ (n - 1)
    //f(n) = 2f(n - 1)
    public int JumpFloorII(int target) {
//        return (int) Math.pow(2,target-1);
        int result = 1;
        while (--target>0){
            result *= 2;
        }
        return result;
    }

    //矩形覆盖
    /* 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
     * 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     */
    //同跳台阶，只是0的时候应该返回0，数列为：0，1，2，3，5，8，13，。。。。。
    public int RectCover(int target) {
        if(target < 4)
            return target;
        int a = 1;
        int b = 1;
        while (target-- > 0) {
            b += a;
            a = b - a;
        }
        return a;
    }

    //二进制中1的个数
    /*输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。*/
    /* 解答：在机器中，整数都是以二进制补码的形式存在的，运算也是，n & (n - 1)是个小技巧，实际上相当把最右边的1 置换成0
     * 比如 7 = 0111,循环为：
     * result = 1,n = 0111 & 0110 = 0110;
     * result = 2,n = 0110 & 0101 = 0100;
     * result = 3,n = 0100 & 0011 = 0000;
     * 比如 -7 = 1111，循环为：
     * result = 1,n = 1111 & 1110 = 1110;
     * result = 2,n = 1110 & 1101 = 1100;
     * result = 3,n = 1100 & 1011 = 1000;
     * result = 4,n = 1000 & 0111 = 0000;
     */
    public int NumberOf1(int n) {
        int result = 0;
        n = n & 0xffffffff;
        while(n != 0){
            result ++;
            n = n & (n - 1);
        }
        return result;
    }

}
