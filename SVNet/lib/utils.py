import enum

# llvm/lib/Target/X86/X86RegisterInfo.td에서 컴파일된
# X86GenRegisterInfo.inc로부터 추출함
class X86:
  # register numbers in llvm
  AH = 1
  AL = 2
  AX = 3
  BH = 4
  BL = 5
  BP = 6
  BPH = 7
  BPL = 8
  BX = 9
  CH = 10
  CL = 11
  CS = 12
  CX = 13
  DF = 14
  DH = 15
  DI = 16
  DIH = 17
  DIL = 18
  DL = 19
  DS = 20
  DX = 21
  EAX = 22
  EBP = 23
  EBX = 24
  ECX = 25
  EDI = 26
  EDX = 27
  EFLAGS = 28
  EIP = 29
  EIZ = 30
  ES = 31
  ESI = 32
  ESP = 33
  FPCW = 34
  FPSW = 35
  FS = 36
  GS = 37
  HAX = 38
  HBP = 39
  HBX = 40
  HCX = 41
  HDI = 42
  HDX = 43
  HIP = 44
  HSI = 45
  HSP = 46
  IP = 47
  MXCSR = 48
  RAX = 49
  RBP = 50
  RBX = 51
  RCX = 52
  RDI = 53
  RDX = 54
  RIP = 55
  RIZ = 56
  RSI = 57
  RSP = 58
  SI = 59
  SIH = 60
  SIL = 61
  SP = 62
  SPH = 63
  SPL = 64
  SS = 65
  SSP = 66
  BND0 = 67
  BND1 = 68
  BND2 = 69
  BND3 = 70
  CR0 = 71
  CR1 = 72
  CR2 = 73
  CR3 = 74
  CR4 = 75
  CR5 = 76
  CR6 = 77
  CR7 = 78
  CR8 = 79
  CR9 = 80
  CR10 = 81
  CR11 = 82
  CR12 = 83
  CR13 = 84
  CR14 = 85
  CR15 = 86
  DR0 = 87
  DR1 = 88
  DR2 = 89
  DR3 = 90
  DR4 = 91
  DR5 = 92
  DR6 = 93
  DR7 = 94
  DR8 = 95
  DR9 = 96
  DR10 = 97
  DR11 = 98
  DR12 = 99
  DR13 = 100
  DR14 = 101
  DR15 = 102
  FP0 = 103
  FP1 = 104
  FP2 = 105
  FP3 = 106
  FP4 = 107
  FP5 = 108
  FP6 = 109
  FP7 = 110
  K0 = 111
  K1 = 112
  K2 = 113
  K3 = 114
  K4 = 115
  K5 = 116
  K6 = 117
  K7 = 118
  MM0 = 119
  MM1 = 120
  MM2 = 121
  MM3 = 122
  MM4 = 123
  MM5 = 124
  MM6 = 125
  MM7 = 126
  R8 = 127
  R9 = 128
  R10 = 129
  R11 = 130
  R12 = 131
  R13 = 132
  R14 = 133
  R15 = 134
  ST0 = 135
  ST1 = 136
  ST2 = 137
  ST3 = 138
  ST4 = 139
  ST5 = 140
  ST6 = 141
  ST7 = 142
  TMM0 = 143
  TMM1 = 144
  TMM2 = 145
  TMM3 = 146
  TMM4 = 147
  TMM5 = 148
  TMM6 = 149
  TMM7 = 150
  XMM0 = 151
  XMM1 = 152
  XMM2 = 153
  XMM3 = 154
  XMM4 = 155
  XMM5 = 156
  XMM6 = 157
  XMM7 = 158
  XMM8 = 159
  XMM9 = 160
  XMM10 = 161
  XMM11 = 162
  XMM12 = 163
  XMM13 = 164
  XMM14 = 165
  XMM15 = 166
  XMM16 = 167
  XMM17 = 168
  XMM18 = 169
  XMM19 = 170
  XMM20 = 171
  XMM21 = 172
  XMM22 = 173
  XMM23 = 174
  XMM24 = 175
  XMM25 = 176
  XMM26 = 177
  XMM27 = 178
  XMM28 = 179
  XMM29 = 180
  XMM30 = 181
  XMM31 = 182
  YMM0 = 183
  YMM1 = 184
  YMM2 = 185
  YMM3 = 186
  YMM4 = 187
  YMM5 = 188
  YMM6 = 189
  YMM7 = 190
  YMM8 = 191
  YMM9 = 192
  YMM10 = 193
  YMM11 = 194
  YMM12 = 195
  YMM13 = 196
  YMM14 = 197
  YMM15 = 198
  YMM16 = 199
  YMM17 = 200
  YMM18 = 201
  YMM19 = 202
  YMM20 = 203
  YMM21 = 204
  YMM22 = 205
  YMM23 = 206
  YMM24 = 207
  YMM25 = 208
  YMM26 = 209
  YMM27 = 210
  YMM28 = 211
  YMM29 = 212
  YMM30 = 213
  YMM31 = 214
  ZMM0 = 215
  ZMM1 = 216
  ZMM2 = 217
  ZMM3 = 218
  ZMM4 = 219
  ZMM5 = 220
  ZMM6 = 221
  ZMM7 = 222
  ZMM8 = 223
  ZMM9 = 224
  ZMM10 = 225
  ZMM11 = 226
  ZMM12 = 227
  ZMM13 = 228
  ZMM14 = 229
  ZMM15 = 230
  ZMM16 = 231
  ZMM17 = 232
  ZMM18 = 233
  ZMM19 = 234
  ZMM20 = 235
  ZMM21 = 236
  ZMM22 = 237
  ZMM23 = 238
  ZMM24 = 239
  ZMM25 = 240
  ZMM26 = 241
  ZMM27 = 242
  ZMM28 = 243
  ZMM29 = 244
  ZMM30 = 245
  ZMM31 = 246
  R8B = 247
  R9B = 248
  R10B = 249
  R11B = 250
  R12B = 251
  R13B = 252
  R14B = 253
  R15B = 254
  R8BH = 255
  R9BH = 256
  R10BH = 257
  R11BH = 258
  R12BH = 259
  R13BH = 260
  R14BH = 261
  R15BH = 262
  R8D = 263
  R9D = 264
  R10D = 265
  R11D = 266
  R12D = 267
  R13D = 268
  R14D = 269
  R15D = 270
  R8W = 271
  R9W = 272
  R10W = 273
  R11W = 274
  R12W = 275
  R13W = 276
  R14W = 277
  R15W = 278
  R8WH = 279
  R9WH = 280
  R10WH = 281
  R11WH = 282
  R12WH = 283
  R13WH = 284
  R14WH = 285
  R15WH = 286
  K0_K1 = 287
  K2_K3 = 288
  K4_K5 = 289
  K6_K7 = 290

  #### define register family
  F_A = [AL, AH, AX, EAX, RAX]
  F_B = [BL, BH, BX, EBX, RBX]
  F_C = [CL, CH, CX, ECX, RCX]
  F_D = [DL, DH, DX, EDX, RDX]
  F_BP = [BPL, BPH, BP, EBP, RBP]
  F_SI = [SIL, SIH, SI, ESI, RSI]
  F_DI = [DIL, DIH, DI, EDI, RDI]
  F_SP = [SPL, SPH, SP, ESP, RSP]
  F_IP = [IP, EIP, RIP]
  F_8 = [R8B, R8BH, R8W, R8WH, R8D, R8]
  F_9 = [R9B, R9BH, R9W, R9WH, R9D, R9]
  F_10 = [R10B, R10BH, R10W, R10WH, R10D, R10]
  F_11 = [R11B, R11BH, R11W, R11WH, R11D, R11]
  F_12 = [R12B, R12BH, R12W, R12WH, R12D, R12]
  F_13 = [R13B, R13BH, R13W, R13WH, R13D, R13]
  F_14 = [R14B, R14BH, R14W, R14WH, R14D, R14]
  F_15 = [R15B, R15BH, R15W, R15WH, R15D, R15]

  # in fact, ST0 ~ ST7 is not allocatable.
  F_FP0 = [MM0, ST0, FP0]
  F_FP1 = [MM1, ST1, FP1]
  F_FP2 = [MM2, ST2, FP2]
  F_FP3 = [MM3, ST3, FP3]
  F_FP4 = [MM4, ST4, FP4]
  F_FP5 = [MM5, ST5, FP5]
  F_FP6 = [MM6, ST6, FP6]
  F_FP7 = [MM7, ST7, FP7]

  F_MM0 = [XMM0, YMM0, ZMM0]
  F_MM1 = [XMM1, YMM1, ZMM1]
  F_MM2 = [XMM2, YMM2, ZMM2]
  F_MM3 = [XMM3, YMM3, ZMM3]
  F_MM4 = [XMM4, YMM4, ZMM4]
  F_MM5 = [XMM5, YMM5, ZMM5]
  F_MM6 = [XMM6, YMM6, ZMM6]
  F_MM7 = [XMM7, YMM7, ZMM7]
  F_MM8 = [XMM8, YMM8, ZMM8]
  F_MM9 = [XMM9, YMM9, ZMM9]
  F_MM10 = [XMM10, YMM10, ZMM10]
  F_MM11 = [XMM11, YMM11, ZMM11]
  F_MM12 = [XMM12, YMM12, ZMM12]
  F_MM13 = [XMM13, YMM13, ZMM13]
  F_MM14 = [XMM14, YMM14, ZMM14]
  F_MM15 = [XMM15, YMM15, ZMM15]
  F_MM16 = [XMM16, YMM16, ZMM16]
  F_MM17 = [XMM17, YMM17, ZMM17]
  F_MM18 = [XMM18, YMM18, ZMM18]
  F_MM19 = [XMM19, YMM19, ZMM19]
  F_MM20 = [XMM20, YMM20, ZMM20]
  F_MM21 = [XMM21, YMM21, ZMM21]
  F_MM22 = [XMM22, YMM22, ZMM22]
  F_MM23 = [XMM23, YMM23, ZMM23]
  F_MM24 = [XMM24, YMM24, ZMM24]
  F_MM25 = [XMM25, YMM25, ZMM25]
  F_MM26 = [XMM26, YMM26, ZMM26]
  F_MM27 = [XMM27, YMM27, ZMM27]
  F_MM28 = [XMM28, YMM28, ZMM28]
  F_MM29 = [XMM29, YMM29, ZMM29]
  F_MM30 = [XMM30, YMM30, ZMM30]
  F_MM31 = [XMM31, YMM31, ZMM31]

  #### all register family for iteration
  F_ALL = [
    F_A, F_B, F_C, F_D,
    F_BP, F_SI, F_DI, F_SP, F_IP,
    F_8, F_9, F_10, F_11, F_12, F_13, F_14, F_15,
    F_FP0, F_FP1, F_FP2, F_FP3, F_FP4, F_FP5, F_FP6, F_FP7,
    F_MM0, F_MM1, F_MM2, F_MM3, F_MM4, F_MM5, F_MM6, F_MM7,
    F_MM8, F_MM9, F_MM10, F_MM11, F_MM12, F_MM13, F_MM14, F_MM15,
    F_MM16, F_MM17, F_MM18, F_MM19, F_MM20, F_MM21, F_MM22, F_MM23,
    F_MM24, F_MM25, F_MM26, F_MM27, F_MM28, F_MM29, F_MM30, F_MM31
  ]

def is_aliased(p1, p2):
  for f in X86.F_ALL:
    if p1 in f and p2 in f:
      return True

  return False

RenumberMap = [-1]*290
for i, f in enumerate(X86.F_ALL):
  for r in f:
    RenumberMap[r] = i

def renumber_reg(r):
  return RenumberMap[r]

  # for i, f in enumerate(X86.F_ALL):
  #   if r in f:
  #     return i

  # return -1
