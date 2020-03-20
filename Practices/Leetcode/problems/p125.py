class Solution:
    def isPalindrome(self, s: str) -> bool:

        def check(char: str) -> bool:
            char = char.lower()
            is_alpha_num = ('a' <= char and char <= 'z') or ('0' <= char and char <= '9')
            return is_alpha_num

        left_i, right_i = -1, len(s)
        while True:
            for left_i in range(left_i + 1, len(s)):
                if check(s[left_i]):
                    break
            for right_i in range(right_i - 1, -1, -1):
                if check(s[right_i]):
                    break
            if left_i >= right_i:
                return True
            if s[left_i].lower() != s[right_i].lower():
                return False


if __name__ == '__main__':
    print(Solution().isPalindrome("A man, a plan, a canal: Panama"))
