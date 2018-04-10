# -*- coding: utf-8 -*-

'''
Assumptions:
1) jo power aur subscript hoga wo waha se hi _{}  and ^{} aayega
2) ÷÷ ise fraction mana hai
3) also correct appearance of two backslash
'''

d = dict();

d['α'] = r'\alpha'
d['β'] = r'\beta'
d['γ'] = r'\gamma'
d['θ'] = r'\theta'
d['η'] = r'\eta'
d['μ'] = r'\mu'
d['λ'] = r'\lambda'
d['π'] = r'\pi'
d['ρ'] = r'\rho'
d['σ'] = r'\sigma'
d['τ'] = r'\tau'
d['δ'] = r'\delta'
d['ϕ'] = r'\phi'
d['≤'] = r'\leq'
d['≥'] = r'\geq'
d['≠'] = r'\neq'
d['÷'] = r'\div'
d['±'] = r'\pm'
d['∑'] = r'\sum'
d['∫'] = r'\int'
d['∏'] = r'\prod'
d['×'] = r'\times'

''' new ones to be considered'''
d['='] = r'='
d['<'] = r'<'
d['>'] = r'>'
d['['] = r'\left['
d[']'] = r'\right]'
#d['sin^{-1}'] = r'\arcsin'
#d['cos^{-1}'] = r'\arccos'
#d['tan^{-1}'] = r'\arctan'
d['sin'] = r'\sin'
d['cos'] = r'\cos'
d['tan'] = r'\tan'
d['cot'] = r'\cot'
d['sec'] = r'\sec'
d['cosec'] = r'\csc'
d['csc'] = r'\csc'
d['log'] = r'\log'
d['Log'] = r'\log'
d['lim'] = r'\lim'
d['∞'] = r'\infty'
d['ω'] = r'\omega'
d['∓'] = r'\mp'
d['∀'] = r'\forall'
d['∃'] = r'\exists'
d['¬'] = r'\neg'
d['÷÷'] = r'÷÷'
# str = 'a÷÷b÷÷c';
#list = []

class Conversion:
    # Constructor to initialize the class variables
    def __init__(self):
        self.top = -1
        # This array is used a stack
        self.array = []
        # Precedence setting
        self.output = []
        self.precedence = {'≤': 1, '≥': 1, '≠': 1, '=': 1, '+': 2, '-': 2, '±': 2, '∓': 2, '÷÷': 3, '÷': 3, '*': 3, '×': 3, '/': 3,
                            '¬': 5}

    # check if the stack is empty
    def isEmpty(self):
        return True if self.top == -1 else False

    # Return the value of the top of the stack
    def peek(self):
        return self.array[-1]

    # Pop the element from the stack
    def pop(self):
        if not self.isEmpty():
            self.top -= 1
            return self.array.pop()
        else:
            return "$"

    # Push the element to the stack
    def push(self, op):
        self.top += 1
        self.array.append(op)

        # A utility function to check is the given character

    # is operand
    def isOperand(self, ch):
        if ch in self.precedence.keys():
            return False;
        elif ch == ')' or ch == '(':
            return False;
        else:
            return True;

    # Check if the precedence of operator is strictly
    # less than top of stack or not
    def notGreater(self, i):
        try:
            a = self.precedence[i]
            b = self.precedence[self.peek()]
            return True if a <= b else False
        except KeyError:
            return False


# also adds minimum no. of bracket required
def postfixToInfix(exp):
    ob = Conversion();
    ob3 = Conversion();
    print(exp)
    c = ''
    print(ob.array)
    for i in exp:
        print(ob.array)
        if ob.isOperand(i):
            ob.push(i);
            ob3.push(7);
        else:
            a = ob.pop();
            b = ob.pop();
            ao = ob3.pop();
            bo = ob3.pop();
            # for those operator which are not in dictionary eg; + *
            if i in d.keys():
                c = d[i];
            else:
                c = i;
            if i == '÷÷':
                c = r'\frac{' + b + '}{' + a + '}';
            else:
                if (ao < ob3.precedence[i]):
                    a = '\left(' + a + r'\right)';
                if (bo < ob3.precedence[i]):
                    b = '\left(' + b + r'\right)';
                c = b +' '+ c + ' ' +a;
            ob.push(c);
            ob3.push(ob3.precedence[i])
    restr="";
    e=0;
    while(e<len(ob.array)):
        restr= restr + ob.array[e];
        e =e+1;

    return restr

    # The main function that converts given infix expression
    # to postfix expression


def infixToPostfix(ob, exp):
    # Iterate over the expression for conversion
    count = 0;
    for i in exp:
        # If the character is an operand,
        # add it to output
        if ob.isOperand(i):
            if (count > 0):
                x = ob.output.pop();
                ob.output.append(x + ' ' +i);
            else:
                ob.output.append(i);
            count = count + 1;

        # If the character is an '(', push it to stack
        elif i == '(':
            count = 0;
            ob.push(i)

        # If the scanned character is an ')', pop and
        # output from the stack until and '(' is found
        elif i == ')':
            count = 0;
            while ((not ob.isEmpty()) and ob.peek() != '('):
                a = ob.pop()
                ob.output.append(a)
            if (not ob.isEmpty() and ob.peek() != '('):
                return -1
            else:
                ob.pop()

        # An operator is encountered
        else:
            count = 0;
            while (not ob.isEmpty() and ob.notGreater(i)):
                ob.output.append(ob.pop())
            ob.push(i)

    # pop all the operator from the stack
    while not ob.isEmpty():
        ob.output.append(ob.pop())

    return ob.output


# Driver program to test above function
def tolatex(st):
    #list.clear()
    list = []
    str = st
     # print(str);
    j = 0
    le = len(str);
    i = 0
    while i < le:
        x = ''
        for j in range(8, 0, -1):
            if str[i:min(i + j, le)].lower() in d.keys():
                # print(d[str[i:min(i + j, le)]]);
                list.append(d[str[i:min(i + j, le)].lower()]);
                i = i + j;
                break;
            elif j == 1:
                if str[i:i + j].lower() in d.keys():
                    x = d[str[i:i + j].lower()];
                else:
                    x = str[i:i + j];
                list.append(x);
                i = i + j;
                break;
    #obj = Conversion();
    #postfix = infixToPostfix(obj, list)
    res = ""
    for i in list:
        if i.lower() in d.keys():
            c = d[i.lower()];
        else:
            c = i
        res = res + c
    #print(list)
    return res

