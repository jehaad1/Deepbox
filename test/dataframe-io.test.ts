/**
 * Parquet and XLSX IO tests.
 *
 * The writers are validated for spec compliance (the parquet output has been
 * verified readable by pyarrow, the xlsx output by openpyxl); the embedded
 * base64 fixtures below were produced by pyarrow (uncompressed PLAIN v1) and
 * openpyxl (deflate zip, inline strings, two sheets) so the readers are
 * exercised against real third-party files without external dependencies.
 */

import { describe, expect, it } from "vitest";
import { DataValidationError } from "../src/core";
import { readParquet, readXlsx, writeParquet, writeXlsx } from "../src/dataframe";

const OPENPYXL_XLSX_B64 =
  "UEsDBBQAAAAIAC5M41xGx01IlQAAAM0AAAAQAAAAZG9jUHJvcHMvYXBwLnhtbE3PTQvCMAwG4L9SdreZih6kDkQ9ip68zy51" +
  "hbYpbYT67+0EP255ecgboi6JIia2mEXxLuRtMzLHDUDWI/o+y8qhiqHke64x3YGMsRoPpB8eA8OibdeAhTEMOMzit7Dp1C5G" +
  "Z3XPlkJ3sjpRJsPiWDQ6sScfq9wcChDneiU+ixNLOZcrBf+LU8sVU57mym/8ZAW/B7oXUEsDBBQAAAAIAC5M41zl3PeH6gAA" +
  "AMsBAAARAAAAZG9jUHJvcHMvY29yZS54bWylkcFqwzAMhl+l+J7IiSFjJs2lZacNBits7GZstQ2NY2NrJH37OVmbbmy3Ha3/" +
  "0ycJ19pL7QI+B+cxUItxNdquj1L7NTsSeQkQ9RGtinki+hTuXbCK0jMcwCt9UgeEkvMKLJIyihRMwswvRnZRGr0o/UfoZoHR" +
  "gB1a7ClCkRdwYwmDjX82zMlCjrFdqGEY8kHMXNqogLenx5d5+aztI6leI2tqo6UOqMiFZrrIn8euhm/F+jL7q4BmlSZIOntc" +
  "s2vyKjbb3QNrSl5WGb/LuNjxSgohy/v3yfWj/ya0zrT79h/Gq6Cp4de/NZ9QSwMEFAAAAAgALkzjXJlcnCMQBgAAnCcAABMA" +
  "AAB4bC90aGVtZS90aGVtZTEueG1s7Vpbc9o4FH7vr9B4Z/ZtC8Y2gba0E3Npdtu0mYTtTh+FEViNbHlkkYR/v0c2EMuWDe2S" +
  "TbqbPAQs6fvORUfn6Dh58+4uYuiGiJTyeGDZL9vWu7cv3uBXMiQRQTAZp6/wwAqlTF61WmkAwzh9yRMSw9yCiwhLeBTL1lzg" +
  "WxovI9bqtNvdVoRpbKEYR2RgfV4saEDQVFFab18gtOUfM/gVy1SNZaMBE1dBJrmItPL5bMX82t4+Zc/pOh0ygW4wG1ggf85v" +
  "p+ROWojhVMLEwGpnP1Zrx9HSSICCyX2UBbpJ9qPTFQgyDTs6nVjOdnz2xO2fjMradDRtGuDj8Xg4tsvSi3AcBOBRu57CnfRs" +
  "v6RBCbSjadBk2PbarpGmqo1TT9P3fd/rm2icCo1bT9Nrd93TjonGrdB4Db7xT4fDronGq9B062kmJ/2ua6TpFmhCRuPrehIV" +
  "teVA0yAAWHB21szSA5ZeKfp1lBrZHbvdQVzwWO45iRH+xsUE1mnSGZY0RnKdkAUOADfE0UxQfK9BtorgwpLSXJDWzym1UBoI" +
  "msiB9UeCIcXcr/31l7vJpDN6nX06zmuUf2mrAaftu5vPk/xz6OSfp5PXTULOcLwsCfH7I1thhyduOxNyOhxnQnzP9vaRpSUy" +
  "z+/5CutOPGcfVpawXc/P5J6MciO73fZYffZPR24j16nAsyLXlEYkRZ/ILbrkETi1SQ0yEz8InYaYalAcAqQJMZahhvi0xqwR" +
  "4BN9t74IyN+NiPerb5o9V6FYSdqE+BBGGuKcc+Zz0Wz7B6VG0fZVvNyjl1gVAZcY3zSqNSzF1niVwPGtnDwdExLNlAsGQYaX" +
  "JCYSqTl+TUgT/iul2v6c00DwlC8k+kqRj2mzI6d0Js3oMxrBRq8bdYdo0jx6/gX5nDUKHJEbHQJnG7NGIYRpu/AerySOmq3C" +
  "EStCPmIZNhpytRaBtnGphGBaEsbReE7StBH8Waw1kz5gyOzNkXXO1pEOEZJeN0I+Ys6LkBG/HoY4SprtonFYBP2eXsNJweiC" +
  "y2b9uH6G1TNsLI73R9QXSuQPJqc/6TI0B6OaWQm9hFZqn6qHND6oHjIKBfG5Hj7lengKN5bGvFCugnsB/9HaN8Kr+ILAOX8u" +
  "fc+l77n0PaHStzcjfWfB04tb3kZuW8T7rjHa1zQuKGNXcs3Ix1SvkynYOZ/A7P1oPp7x7frZJISvmlktIxaQS4GzQSS4/IvK" +
  "8CrECehkWyUJy1TTZTeKEp5CG27pU/VKldflr7kouDxb5OmvoXQ+LM/5PF/ntM0LM0O3ckvqtpS+tSY4SvSxzHBOHssMO2c8" +
  "kh22d6AdNfv2XXbkI6UwU5dDuBpCvgNtup3cOjiemJG5CtNSkG/D+enFeBriOdkEuX2YV23n2NHR++fBUbCj7zyWHceI8qIh" +
  "7qGGmM/DQ4d5e1+YZ5XGUDQUbWysJCxGt2C41/EsFOBkYC2gB4OvUQLyUlVgMVvGAyuQonxMjEXocOeXXF/j0ZLj26ZltW6v" +
  "KXcZbSJSOcJpmBNnq8reZbHBVR3PVVvysL5qPbQVTs/+Wa3InwwRThYLEkhjlBemSqLzGVO+5ytJxFU4v0UzthKXGLzj5sdx" +
  "TlO4Ena2DwIyubs5qXplMWem8t8tDAksW4hZEuJNXe3V55ucrnoidvqXd8Fg8v1wyUcP5TvnX/RdQ65+9t3j+m6TO0hMnHnF" +
  "EQF0RQIjlRwGFhcy5FDukpAGEwHNlMlE8AKCZKYcgJj6C73yDLkpFc6tPjl/RSyDhk5e0iUSFIqwDAUhF3Lj7++TaneM1/os" +
  "gW2EVDJk1RfKQ4nBPTNyQ9hUJfOu2iYLhdviVM27Gr4mYEvDem6dLSf/217UPbQXPUbzo5ngHrOHc5t6uMJFrP9Y1h75Mt85" +
  "cNs63gNe5hMsQ6R+wX2KioARq2K+uq9P+SWcO7R78YEgm/zW26T23eAMfNSrWqVkKxE/Swd8H5IGY4xb9DRfjxRiraaxrcba" +
  "MQx5gFjzDKFmON+HRZoaM9WLrDmNCm9B1UDlP9vUDWj2DTQckQVeMZm2NqPkTgo83P7vDbDCxI7h7Yu/AVBLAwQUAAAACAAu" +
  "TONcDRHs/EcBAAA9AgAAGAAAAHhsL3dvcmtzaGVldHMvc2hlZXQxLnhtbE1S226DMAz9FZQPaCjSLqogUtdp2h4mVa22Padg" +
  "IGouLHHL9vdz0pbxgOLj+BwfO5Sj88fQA2D2Y7QNFesRhxXnoe7ByLBwA1i6aZ03Egn6jofBg2wSyWhe5Pk9N1JZJsqU23pR" +
  "uhNqZWHrs3AyRvrfJ9BurNiS3RI71fWYElyUg+xgD/gxEIEgn3QaZcAG5Wzmoa3YerlaF4mRKj4VjGEWZ3GYg3PHCN6aiuXR" +
  "E2ioMUpIOs6wAa2jEjn5voqy/6aROY9v8i9pfrJ3kAE2Tn+pBvuKPbKsgVaeNO7c+ArXme7+LT5LlKL0bsx8HFaUdQxiSypU" +
  "Ni5pj57yijqhkCVHah8Br+kj4sQuJnaR2HHhZ7Es+XlezGdt41rfpe+UDZmGljj54oHM+YvPC0A3pGc4OERnUtjT84KPBXTf" +
  "OocTiHua/hjxB1BLAwQUAAAACAAuTONcBKU600cBAAA9AgAAGAAAAHhsL3dvcmtzaGVldHMvc2hlZXQyLnhtbE1S226DMAz9" +
  "FZQPaCjSLqogUtdp2h4mVa22PQcwEDUXlrhl+/s5acv6gOLj+BwfO5ST84cwAGD2Y7QNFRsQxxXnoRnAyLBwI1i66Zw3Egn6" +
  "nofRg2wTyWhe5Pk9N1JZJsqU23pRuiNqZWHrs3A0RvrfJ9BuqtiSXRM71Q+YElyUo+xhD/gxEoEgn3VaZcAG5WzmoavYerla" +
  "F4mRKj4VTOEmzuIwtXOHCN7aiuXRE2hoMEpIOk6wAa2jEjn5voiy/6aReRtf5V/S/GSvlgE2Tn+pFoeKPbKshU4eNe7c9AqX" +
  "me7+LT5LlKL0bsp8HFaUTQxiSypUNi5pj57yijqhqEuO1D4C3tBHxJldzOwisePCT6Io+em2mN+0jWt9l75XNmQaOuLkiwcy" +
  "588+zwDdmJ6hdojOpHCg5wUfC+i+cw5nEPc0/zHiD1BLAwQUAAAACAAuTONc0gXxRlICAABHCgAADQAAAHhsL3N0eWxlcy54" +
  "bWzdVtuK2zAQ/RXjD6iTmJq4JHmoIVBoy8LuQ1/lWE4EuriyvCT9+mok57ab41L6VpvgmTk6M2ekMc6qdyfJnw+cu+SopO7X" +
  "6cG57lOW9bsDV6z/YDquPdIaq5jzrt1nfWc5a3oiKZktZrMiU0zodLPSg9oq1yc7M2i3Tmdpkm1WrdHX0DyNAb+WKZ68MrlO" +
  "KyZFbUVczJSQpxhfhMjOSGMT59VwolOo/xUXzEeXpI65lNDGhmgWy4RH7xMLKS8qFmkMbFYdc45bvfVOJIXoe2y0X06dV7G3" +
  "7DRffExvGOHhy9TGNtzetRtDm5XkrSOGFftDMJzp6FEb54wiqxFsbzSLSs600fC5d1zKZzqvH+1dgWObxI3/0oQ9p47Pplc1" +
  "mjHN6FCB23Qx+b/n7cSrcZ8H35AO/s/BOP5keSuOwT+2bwRcagcld+Uv0YRGZZ1+pxGUNznqQUgn9OgdRNNw/b47n9+x2g/5" +
  "XQG/quEtG6R7uYDr9Gp/440YVHlZ9USNjauu9lc6ynlxnVNfTOiGH3lTja7d18FMvOHLjldgvIW24QIQZEUQQATCWlAGZEUe" +
  "rPU/9rXEfUUQKlw+hpaYtcSsyHsIVeGGtQCr9BdouSzzvCjg9lbVYxkV3MOioB9ICBUSB9aian+78xMDMDE2f5gNeMqTYwNb" +
  "nhhR2PLEzhME9pA4ZQkGANYiDjwUOFEkAtSiUQOsPKdzhgrhaz4BlSWEaEjB9BYF2qiCbnBe8CXK87IEEIFARp5DiF7YCQjK" +
  "ICEQyvP4IX3zPcvO37ns+tdx8xtQSwMEFAAAAAgALkzjXLdH64rAAAAAFgIAAAsAAABfcmVscy8ucmVsc52SS24CMQxArxJl" +
  "X0ypxAIxrNiwQ4gLuInno5nEkWPE9PaN2MAgaBFL/56eLa8PNKB2HHPbpWzGMMRc2VY1rQCyaylgnnGiWCo1S0AtoTSQ0PXY" +
  "ECzm8yXILcNu1rdMc/xJ9AqR67pztGV3ChT1Afiuw5ojSkNa2XGAM0v/zdzPCtSana+s7PynNfCmzPP1IJCiR0VwLPSRpEyL" +
  "dpSvPp7dvqTzpWNitHjf6P/z0KgUPfm/nTClidLXRQkmb7D5BVBLAwQUAAAACAAuTONcGnnHhEEBAABnAgAADwAAAHhsL3dv" +
  "cmtib29rLnhtbI2RYUvDMBCG/0rJD7Bd0YFj9YtjOhAVJ/ueJdf1WJIryXXT/XqTlmpBED+l997d0/dNlmfyxz3RMfuwxoWF" +
  "r0TD3C7yPKgGrAxX1IKLvZq8lRxLf8iprlHBilRnwXFeFsU892AkI7nQYBvEQPsPK7QepA4NAFszoKxEJ+6Wo7NXn+XTihhU" +
  "+lNSk7JDOIefgVRmJwy4R4P8WYn+24DILDq0eAFdiUJkoaHzI3m8kGNptsqTMZWYDY0deEb1S94mm+9yH3qF5f4tZa7EvIjA" +
  "Gn3gfqLny2jyBHF4qDqmNRoGv5IMD566Ft2hx8QY+SRHfxXjmTlpoRLrhE4WorTRgx2OnEk4v8DY8Bs9EKfbW1Dk9GS9/GO9" +
  "HAyNLjTU6EA/R1BIjXgnKj5IOnoj5fXN7DZm74y5j9qLeyKpv2ONb3L3BVBLAwQUAAAACAAuTONcq15yLrQAAACNAgAAGgAA" +
  "AHhsL19yZWxzL3dvcmtib29rLnhtbC5yZWxzxZJNCoMwEEavEnIAR23poqirbtwWLxB0/MHEhMyU6u0rulChi26kq/BNyPse" +
  "TJInasWdHajtHInR6IFS2TK7OwCVLRpFgXU4zDe19UbxHH0DTpW9ahDiMLyB3zNkluyZopgc/kK0dd2V+LDly+DAX8Dwtr6n" +
  "FpGlKJRvkFMJo97GBMsRBTNZirxKpc+rSAr4t1F8MIrPNCKeNNKms+ZD/+XMfp7f4la/xHV4XMt1kYDD78s+UEsDBBQAAAAI" +
  "AC5M41yl4RtYHwEAAGAEAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbMVUy07DMBD8lcjXKnbpgQNqeqFcoQd+wCSbxopf8m5L" +
  "+vdsEloJVFqqIHGJFe/szHjH8vL1EAGzzlmPhWiI4oNSWDbgNMoQwXOlDslp4t+0VVGXrd6CWszn96oMnsBTTj2HWC3XUOud" +
  "peyp4200wRcigUWRPY7AXqsQOkZrSk1cV3tffVPJPxUkdw4YbEzEGQNEps5KDKUfFY6NL3tIyVSQbXSiZ+0YpjqrkA4WUF7m" +
  "OOMy1LUpoQrlznGLxJhAV9gAkLNyJJ1dkSYeMozfu8kGBpqLigzdpBCRU0twu94xlr47j0wEicyVQ54kmXvyCaFPvILqt+I8" +
  "4feQ2iETVMMyfcxfcz7x32pk8Z9G3kJo//rC96t02viTATU8LKsPUEsBAhQDFAAAAAgALkzjXEbHTUiVAAAAzQAAABAAAAAA" +
  "AAAAAAAAAIABAAAAAGRvY1Byb3BzL2FwcC54bWxQSwECFAMUAAAACAAuTONc5dz3h+oAAADLAQAAEQAAAAAAAAAAAAAAgAHD" +
  "AAAAZG9jUHJvcHMvY29yZS54bWxQSwECFAMUAAAACAAuTONcmVycIxAGAACcJwAAEwAAAAAAAAAAAAAAgAHcAQAAeGwvdGhl" +
  "bWUvdGhlbWUxLnhtbFBLAQIUAxQAAAAIAC5M41wNEez8RwEAAD0CAAAYAAAAAAAAAAAAAACAgR0IAAB4bC93b3Jrc2hlZXRz" +
  "L3NoZWV0MS54bWxQSwECFAMUAAAACAAuTONcBKU600cBAAA9AgAAGAAAAAAAAAAAAAAAgIGaCQAAeGwvd29ya3NoZWV0cy9z" +
  "aGVldDIueG1sUEsBAhQDFAAAAAgALkzjXNIF8UZSAgAARwoAAA0AAAAAAAAAAAAAAIABFwsAAHhsL3N0eWxlcy54bWxQSwEC" +
  "FAMUAAAACAAuTONct0frisAAAAAWAgAACwAAAAAAAAAAAAAAgAGUDQAAX3JlbHMvLnJlbHNQSwECFAMUAAAACAAuTONcGnnH" +
  "hEEBAABnAgAADwAAAAAAAAAAAAAAgAF9DgAAeGwvd29ya2Jvb2sueG1sUEsBAhQDFAAAAAgALkzjXKteci60AAAAjQIAABoA" +
  "AAAAAAAAAAAAAIAB6w8AAHhsL19yZWxzL3dvcmtib29rLnhtbC5yZWxzUEsBAhQDFAAAAAgALkzjXKXhG1gfAQAAYAQAABMA" +
  "AAAAAAAAAAAAAIAB1xAAAFtDb250ZW50X1R5cGVzXS54bWxQSwUGAAAAAAoACgCEAgAAJxIAAAAA";

const PYARROW_PARQUET_B64 =
  "UEFSMRUAFRwVHCwVBhUAFQYVBhwYBAMAAAAYBAEAAAAWAigEAwAAABgEAQAAABERAAAAAgAAAAMFAQAAAAMAAAAVABUgFSAs" +
  "FQYVABUGFQYcNgIoAWMYAWEREQAAAAIAAAADBQEAAABhAQAAAGMVABUOFQ4sFQYVABUGFQYcGAEBGAEAFgIoAQEYAQAREQAA" +
  "AAIAAAADAwEVABUcFRwsFQYVABUGFQYcGAQAAKA/GAQAAAA/FgIoBAAAoD8YBAAAAD8REQAAAAIAAAADBQAAAD8AAKA/FQIZ" +
  "XDUAGAZzY2hlbWEVCAAVAiUCGAF4ABUMJQIYAXMlAEwcAAAAFQAlAhgEZmxhZwAVCCUCGANmMzIAFgYZHBlMJgAcFQIZJQYA" +
  "GRgBeBUAFgYWehZ6Jgg8GAQDAAAAGAQBAAAAFgIoBAMAAAAYBAEAAAAREQAZHBUAFQAVAgA8KQYZJgIEAAAAJgAcFQwZJQYA" +
  "GRgBcxUAFgYWWhZaJoIBPDYCKAFjGAFhEREAGRwVABUAFQIAPBYEGQYZJgIEAAAAJgAcFQAZJQYAGRgEZmxhZxUAFgYWVBZU" +
  "JtwBPBgBARgBABYCKAEBGAEAEREAGRwVABUAFQIAPCkGGSYCBAAAACYAHBUIGSUGABkYA2YzMhUAFgYWehZ6JrACPBgEAACg" +
  "PxgEAAAAPxYCKAQAAKA/GAQAAAA/EREAGRwVABUAFQIAPCkGGSYCBAAAABaiAxYGJggWogMAGRwYDEFSUk9XOnNjaGVtYRjs" +
  "Ai8vLy8vd2dCQUFBUUFBQUFBQUFLQUF3QUJnQUZBQWdBQ2dBQUFBQUJCQUFNQUFBQUNBQUlBQUFBQkFBSUFBQUFCQUFBQUFR" +
  "QUFBQ2dBQUFBWkFBQUFEZ0FBQUFFQUFBQWdQLy8vd0FBQVFNUUFBQUFIQUFBQUFRQUFBQUFBQUFBQXdBQUFHWXpNZ0FBQUFZ" +
  "QUNBQUdBQVlBQUFBQUFBRUFzUC8vL3dBQUFRWVFBQUFBR0FBQUFBUUFBQUFBQUFBQUJBQUFBR1pzWVdjQUFBQUEzUC8vLzlq" +
  "Ly8vOEFBQUVGRUFBQUFCZ0FBQUFFQUFBQUFBQUFBQUVBQUFCekFBQUFCQUFFQUFRQUFBQVFBQlFBQ0FBR0FBY0FEQUFBQUJB" +
  "QUVBQUFBQUFBQVFJUUFBQUFIQUFBQUFRQUFBQUFBQUFBQVFBQUFIZ0FBQUFJQUF3QUNBQUhBQWdBQUFBQUFBQUJJQUFBQUFB" +
  "QUFBQT0AGCBwYXJxdWV0LWNwcC1hcnJvdyB2ZXJzaW9uIDI0LjAuMBlMHAAAHAAAHAAAHAAAAAoDAABQQVIx";
const fromB64 = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));

describe("parquet writer/reader round-trip", () => {
  it("round-trips all supported types exactly", () => {
    const cols = ["i", "f", "s", "b", "big"];
    const data = [
      { i: 1, f: 1.5, s: "hello", b: true, big: 10n },
      { i: -2, f: -2.25, s: "wörld ✓", b: false, big: -5n },
    ];
    const out = readParquet(writeParquet(cols, data));
    expect(out.columns).toEqual(cols);
    expect(out.data).toEqual([
      { i: 1, f: 1.5, s: "hello", b: true, big: 10 },
      { i: -2, f: -2.25, s: "wörld ✓", b: false, big: -5 },
    ]);
  });

  it("preserves nulls via OPTIONAL columns with definition levels", () => {
    const cols = ["i", "f", "s", "b"];
    const data = [
      { i: 1, f: 0.5, s: "x", b: true },
      { i: null, f: null, s: null, b: null },
      { i: 3, f: 1.5, s: "z", b: false },
    ];
    const out = readParquet(writeParquet(cols, data));
    expect(out.data[1]).toEqual({ i: null, f: null, s: null, b: null });
    expect(out.data[0]).toEqual({ i: 1, f: 0.5, s: "x", b: true });
    expect(out.data[2]).toEqual({ i: 3, f: 1.5, s: "z", b: false });
  });

  it("infers types from the whole column (no int truncation)", () => {
    // 1.5 appears after integers: the column must be DOUBLE, not INT32.
    const out = readParquet(writeParquet(["v"], [{ v: 1 }, { v: 2 }, { v: 1.5 }]));
    expect(out.data.map((r) => r.v)).toEqual([1, 2, 1.5]);
    // Integers beyond int32 range use INT64.
    const big = readParquet(writeParquet(["v"], [{ v: 1 }, { v: 2 ** 40 }]));
    expect(big.data.map((r) => r.v)).toEqual([1, 2 ** 40]);
  });

  it("bit-packs booleans (odd counts exercise the tail)", () => {
    const rows = Array.from({ length: 11 }, (_, i) => ({ ok: i % 3 === 0 }));
    const out = readParquet(writeParquet(["ok"], rows));
    expect(out.data.map((r) => r.ok)).toEqual(rows.map((r) => r.ok));
  });

  it("honors the columns option", () => {
    const buf = writeParquet(["a", "b"], [{ a: 1, b: "x" }]);
    const out = readParquet(buf, { columns: ["b"] });
    expect(out.columns).toEqual(["b"]);
    expect(out.data).toEqual([{ b: "x" }]);
  });

  it("handles empty tables and rejects invalid buffers", () => {
    const empty = readParquet(writeParquet(["a"], []));
    expect(empty.columns).toEqual(["a"]);
    expect(empty.data).toEqual([]);

    // Not a Parquet file: throw instead of returning an empty table.
    expect(() => readParquet(new Uint8Array(4))).toThrow(DataValidationError);
    expect(() => readParquet(new Uint8Array(64))).toThrow(DataValidationError);
  });

  it("reads a pyarrow-written uncompressed PLAIN v1 file, nulls included", () => {
    const out = readParquet(fromB64(PYARROW_PARQUET_B64));
    expect(out.columns).toEqual(["x", "s", "flag", "f32"]);
    expect(out.data).toEqual([
      { x: 1, s: "a", flag: true, f32: 0.5 },
      { x: null, s: null, flag: false, f32: null },
      { x: 3, s: "c", flag: null, f32: 1.25 },
    ]);
  });
});

describe("xlsx writer/reader round-trip", () => {
  it("round-trips numbers, booleans, strings and unescapes XML entities", () => {
    const cols = ["a", "b <x> & 'q'"];
    const buf = writeXlsx(cols, [
      { a: 1.5, "b <x> & 'q'": "he&<>llo" },
      { a: true, "b <x> & 'q'": "plain" },
    ]);
    const out = readXlsx(buf);
    expect(out.columns).toEqual(cols);
    expect(out.data).toEqual([
      { a: 1.5, "b <x> & 'q'": "he&<>llo" },
      { a: true, "b <x> & 'q'": "plain" },
    ]);
  });

  it("writes nulls as empty cells and reads them back as null", () => {
    const buf = writeXlsx(
      ["a", "b"],
      [
        { a: null, b: 2 },
        { a: "x", b: undefined },
      ]
    );
    expect(readXlsx(buf).data).toEqual([
      { a: null, b: 2 },
      { a: "x", b: null },
    ]);
  });

  it("supports header:false and many columns (AA+ refs)", () => {
    const cols = Array.from({ length: 30 }, (_, i) => `c${i}`);
    const row: Record<string, number> = {};
    for (let i = 0; i < 30; i++) row[`c${i}`] = i;
    const buf = writeXlsx(cols, [row]);

    const withHeader = readXlsx(buf);
    expect(withHeader.columns).toEqual(cols);
    expect(withHeader.data[0]?.c29).toBe(29);

    const noHeader = readXlsx(buf, { header: false });
    expect(noHeader.columns[0]).toBe("Column0");
    expect(noHeader.data).toHaveLength(2); // header row becomes data
    expect(noHeader.data[1]?.Column29).toBe(29);
  });

  it("uses the sheetName option when writing", () => {
    const buf = writeXlsx(["a"], [{ a: 1 }], { sheetName: "Custom" });
    expect(readXlsx(buf, { sheet: "Custom" }).data).toEqual([{ a: 1 }]);
  });

  it("throws for non-zip buffers instead of returning an empty sheet", () => {
    expect(() => readXlsx(new Uint8Array(16))).toThrow(DataValidationError);
  });

  it("reads an openpyxl file (deflate zip, inline strings, multiple sheets)", () => {
    const buf = fromB64(OPENPYXL_XLSX_B64);
    expect(readXlsx(buf)).toEqual({ columns: ["a"], data: [{ a: 1 }] });
    expect(readXlsx(buf, { sheet: "Second" })).toEqual({ columns: ["b"], data: [{ b: 2 }] });
    expect(() => readXlsx(buf, { sheet: "Nope" })).toThrow(DataValidationError);
    expect(() => readXlsx(buf, { sheet: "Nope" })).toThrow(/available sheets: First, Second/);
  });
});

describe("xlsx decompression-bomb guard", () => {
  it("rejects entries that decompress beyond maxEntryBytes", async () => {
    // A ~50 MB run of zeros compresses to a few KB; a tight cap must reject it.
    const { deflateRawSync } = await import("node:zlib");
    const huge = deflateRawSync(Buffer.alloc(50 * 1024 * 1024));

    // Hand-build a tiny zip containing the bomb as sharedStrings.xml.
    const name = Buffer.from("xl/sharedStrings.xml");
    const local = Buffer.concat([
      Buffer.from([0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0]),
      Buffer.alloc(12), // crc + sizes (crc unchecked by reader)
      Buffer.from([name.length, 0, 0, 0]),
      name,
      huge,
    ]);
    // Patch compressed size into the local header (offset 18) and central dir.
    local.writeUInt32LE(huge.length, 18);
    const central = Buffer.concat([
      Buffer.from([0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0]),
      Buffer.alloc(12),
      Buffer.from([name.length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      Buffer.alloc(4), // local header offset = 0
      name,
    ]);
    central.writeUInt32LE(huge.length, 20);
    const eocd = Buffer.concat([
      Buffer.from([0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 1, 0, 1, 0]),
      Buffer.alloc(10),
    ]);
    eocd.writeUInt32LE(central.length, 12);
    eocd.writeUInt32LE(local.length, 16);
    const zip = new Uint8Array(Buffer.concat([local, central, eocd]));

    expect(() => readXlsx(zip, { maxEntryBytes: 1024 * 1024 })).toThrow(/decompression bomb/);
    // With a sufficient cap the same archive parses (no sheets -> empty result).
    expect(readXlsx(zip, { maxEntryBytes: 100 * 1024 * 1024 })).toEqual({ columns: [], data: [] });
  });
});
