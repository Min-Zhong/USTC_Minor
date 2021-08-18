.data
    sortarray:
        .space 40
    separate:
        .asciiz " "
    line:
        .asciiz "\n"

.text
.globl main

main:
    la $t0, sortarray          
    add $t1, $zero, $t0        
    addi $t8, $t0, 40          

    addi $t3, $zero, 0         
    inputData:
        li $v0, 5              
        syscall
        sw $v0, 0($t1)         

        addi $t1, $t1, 4       
        addi $t3, $t3, 1       
        slti $s0, $t3, 10      
        bnez $s0, inputData

        addi $t3, $zero, 0     
    outLoop:
        add $t1, $zero, $t0    
        slti $s0, $t3, 10      
        beqz $s0, print        

        addi $t4, $t3, -1      
    inLoop:
        slti $s0, $t4, 0       
        bnez $s0, exitInLoop

        sll $t5, $t4, 2        
        add $t5, $t1, $t5      
        lw $t6, 0($t5)         
        lw $t7, 4($t5)         
        slt $s0, $t6, $t7      
        bnez $s0, swap
        addi $t4, $t4, -1      
        j inLoop               

    swap:
        sw $t6, 4($t5)         
        sw $t7, 0($t5)         
        addi $t4, $t4, -1      
        j inLoop               

    exitInLoop:
        addi $t3, $t3, 1       
        j outLoop              

    print:
        lw $a0, 0($t0)         
        li $v0, 1              
        syscall

        la $a0, separate       
        li $v0, 4              
        syscall

        addi $t0, $t0, 4       
        bne $t0, $t8, print    

        la $a0, line           
        li $v0, 4              
        syscall

        j exit                 

    exit:
        li $v0, 10             
        syscall
