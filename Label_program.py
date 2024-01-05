import os
inicial = os.getcwd()

class program:
    def __init__(self):
        pass

    def menu_opcoes1(self):
        print('Renomeação de arquivo')
        print('======================')
        print('Diretório: {}\n'.format(atual))
        print('Escolha uma opção:')
        print('0 - Selecionar uma Pasta\n1 - Renomear um arquivo')
        print('====================== \n')
        print('Outras opções:')
        if atual == inicial:
            print('X - Encerrar')
            print('\n------------------------')
        elif atual != inicial:
            print('V - Voltar \nX - Encerrar\n')
            print('\n------------------------')

    def menu_opcoes2(self):
        print('Renomeação de arquivo')
        print('======================')
        print('Diretório: {}\n'.format(atual))
        print('Selecione uma pasta:')
        for elemento in lista:
            print(elemento)
        print('====================== \n')
        print('Outras opções:')
        if atual == inicial:
            print('X - Encerrar')
            print('\n------------------------')
        elif atual != inicial:
            print('V - Voltar \nX - Encerrar\n')
            print('\n------------------------')
    
    def menu_opcoes3(self):
        print('Renomeação de arquivo')
        print('======================')
        print('Informe um nome base:')
        
    
    def menu_aviso(self):
        print('Renomeação de arquivo')
        print('======================')
        print('AVISO!!')
        print('A ferramenta irá renomear permanentemente todos os arquivos desta pasta,\nmantendo a extensão dos mesmo!\n\
Será necessário indicar um nome base e a ferramenta irá renomeá-lo com este nome \nadicionando uma sequência numérica!')
        print('\n')
        print('Exemplo: ----------------------')
        print('Nome antigo: Cg_cable_exten.jpg')
        print('Se escolhermos o nome base "cable" e este arquivo estiver na posição 5 na pasta,\n\
a ferramenta irá renomear o arquivo da seguinte forma:')
        print('Novo nome: cable_5.jpg')
        print('------------------------------')
        print('\n')

        print('Deseja Continuar?')
        print('0 - Sim\n1 - Não\n')
        print('\n------------------------')
    
    def menu_sem_pasta(self):
        print('Não Existe Pasta neste diretório!!')
        print('\n')
        print('Você deseja voltar?')
        print('0 - Sim\n1 - Encerrar\n')
        print('\n------------------------')
    
    def menu_conclusao(self):
        print('CONCLUÍDO !!')
        print('\n')
        print('Você deseja continuar aplicando a ferramenta?')
        print('0 - Sim\n1 - Encerrar\n')
        print('\n------------------------')
    
    def encerra (self):
        print('==========================')
        print('Aplicação finalizada!\nAté logo!')
        print('==========================')

class proc_nome:
    def __ini__(self):
        pass
    def renomear(self):
        global caminho_antigo
        lista_arq = []
        # 
        for i in range(len(opcoes)):
            if os.path.isdir(opcoes[i]) == False:
                lista_arq.append(opcoes[i])
        caminho_antigo = [os.path.join(atual, i) for i in ((lista_arq))]
        #
        caminho_novo = []
        for posi in range((len(lista_arq))):
            p_nome = lista_arq[posi].split('.')[0] = nome_base + str(posi + 1)
            lista_arq[posi] = p_nome + "." + lista_arq[posi].split('.')[-1]

            caminho_novo.append(os.path.join(atual, lista_arq[posi]))
            os.rename(caminho_antigo[posi],caminho_novo[posi])  

programa = program()
renomea = proc_nome()
while True:
    os.system("cls") or None
    atual = os.getcwd()
    opcoes = os.listdir()

    lista_op = []
    for i in range(len(opcoes)):
        if os.path.isdir(opcoes[i]) == True:
            lista_op.append(opcoes[i])

    lista = [str(i) + " - " + lista_op[i] for i in range(len(lista_op))]    

    ## Menu de opções 
    for i in range(len(opcoes)):
        if os.path.isdir(opcoes[i]) == True:
            valor = 1                                         
        else:
            valor = 0
            break
    if valor == 0:
        programa.menu_opcoes1()
        sel = input()

        if sel.upper() == 'X':
            os.system("cls") or None
            programa.encerra()
            break

        elif int(sel) == 0:
            lista_verific = []
            for i in range(len(opcoes)):
                lista_verific.append(os.path.isdir(opcoes[i]) == True)
            if True not in lista_verific:
                os.system("cls") or None
                programa.menu_sem_pasta()
                verifica = input()
                if int(verifica) == 0:
                    voltar = os.getcwd().split('\\')[:-1]
                    voltar = '//'.join(voltar)
                    os.chdir(voltar)
                elif int(verifica)== 1:
                    break
            else:
                os.system("cls") or None
                programa.menu_opcoes2()
                inp = input()
                ## Verificação de condição
                if inp.upper() == 'V':
                    voltar = os.getcwd().split('\\')[:-1]
                    voltar = '//'.join(voltar)
                    os.chdir(voltar)
                elif inp.upper() == 'X':
                    break
                else:
                    os.chdir(os.path.join(atual,lista_op[int(inp)]))

        elif int(sel) == 1: 
            os.system("cls") or None
            ## RENOMEANDO OS ARQUIVOS
            programa.menu_aviso()
            aviso = input()
            if int(aviso) == 0:
                os.system("cls") or None
                programa.menu_opcoes3()
                nome_base = input()+"_"
                renomea.renomear()

                os.system("cls") or None
                programa.menu_conclusao()
                conclusao = input()
                if int(conclusao) == 0:
                    os.chdir(inicial)
                else:
                    os.system("cls") or None
                    programa.encerra()
                    break
            else:
                voltar = os.getcwd().split('\\')[:-1]
                voltar = '//'.join(voltar)
                os.chdir(voltar)


    else:
        programa.menu_opcoes2()
        inp = input()
        ## Verificação de condição
        if inp.upper() == 'V':
            voltar = os.getcwd().split('\\')[:-1]
            voltar = '//'.join(voltar)
            os.chdir(voltar)
        elif inp.upper() == 'X':
            os.system("cls") or None
            programa.encerra()
            break
        else:
            os.chdir(os.path.join(atual,lista_op[int(inp)]))
