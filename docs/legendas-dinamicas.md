# Legendas dinâmicas (fork Eric-Henrique/Concatericdatelion)

O que esta versão acrescenta à janela **Legendas** do Concat:

| Opção | O que faz |
|---|---|
| **Estilo** | Escolhe o visual das legendas entre "Seu estilo de título" (o guardado por *Usar em novos títulos*) e qualquer estilo de texto, incluindo oito estilos novos de legenda: *Viral amarelo*, *Caixa preta*, *Caixa branca*, *Caixa vermelha*, *Neon verde*, *Gibi*, *Limpo* e *Bebas*. |
| **Palavras** | *Frase* (como antes) ou **1, 2 ou 3 palavras por vez**, no tempo de cada palavra. A legenda quebra antes do limite onde a frase termina, numa vírgula ou numa pausa da fala. |
| **Animação** | Entrada de cada legenda: *Nenhuma*, *Pop* (cresce e assenta), *Fade* ou *Subir*. Vira keyframes normais no título, editáveis depois. |
| **Maiúsculas** | Todas as legendas em CAIXA ALTA. |

As escolhas ficam salvas e voltam na próxima vez que a janela abrir.

## Como usar

1. Em **Configurações › Transcrição**, baixe um modelo, de preferência o **Small (Multilingual)** para português.
2. Selecione o clipe com a fala e abra **Legendas**. Ligue *Usar a voz da linha do tempo*.
3. Escolha **Estilo**, **Palavras** (2 ou 3 é o visual de Reels/TikTok), **Animação** e, se quiser, **Maiúsculas**.
4. Clique em **Gerar legendas**. Tudo entra como uma única ação: um Ctrl+Z desfaz o lote inteiro.

Também vale para um roteiro digitado: o tempo de cada palavra é estimado pelo tamanho dela.

## Como funciona

- O Whisper (whisper.cpp) agora devolve o tempo de cada palavra (*token timestamps*). Quando ele não consegue cronometrar uma frase, as palavras são distribuídas pelo tempo da frase de acordo com o tamanho de cada uma.
- Intervalos curtos entre legendas são fechados, para o texto não piscar. As legendas ficam lado a lado na mesma faixa.
- A animação é feita com keyframes de escala, opacidade e posição em cada título. O comando `AddTextClip` passou a aceitar `keys` para isso acontecer num único lote.

## Build

O workflow `.github/workflows/build-windows.yml` compila o instalador do Windows a cada push na branch `legendas-dinamicas` e publica o `setup.exe` e o `.msi` em **Releases** deste fork.
