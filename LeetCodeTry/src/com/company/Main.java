package com.company;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        Solution solution = new Solution();
        Solution.ListNode a1 = new Solution.ListNode(1);
        Solution.ListNode a2 = new Solution.ListNode(2);
        Solution.ListNode a3 = new Solution.ListNode(3);
        a1.next = a2;
        a2.next = a3;
        System.out.println(solution.FindKthToTail(a1,1).val);

    }
}
