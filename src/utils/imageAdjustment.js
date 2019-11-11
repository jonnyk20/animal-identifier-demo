/* eslint-disable */
export const getImageOrientation = (file, callback) => {
  var reader = new FileReader()
  reader.onload = e => {
    var view = new DataView(e.target.result)
    if (view.getUint16(0, false) != 0xffd8) {
      return callback(-2)
    }
    var length = view.byteLength,
      offset = 2
    while (offset < length) {
      if (view.getUint16(offset + 2, false) <= 8) return callback(-1)
      var marker = view.getUint16(offset, false)
      offset += 2
      if (marker == 0xffe1) {
        if (view.getUint32((offset += 2), false) != 0x45786966) {
          return callback(-1)
        }

        var little = view.getUint16((offset += 6), false) == 0x4949
        offset += view.getUint32(offset + 4, little)
        var tags = view.getUint16(offset, little)
        offset += 2
        for (var i = 0; i < tags; i++) {
          if (view.getUint16(offset + i * 12, little) == 0x0112) {
            return callback(view.getUint16(offset + i * 12 + 8, little))
          }
        }
      } else if ((marker & 0xff00) != 0xff00) {
        break
      } else {
        offset += view.getUint16(offset, false)
      }
    }
    return callback(0)
  }
  reader.readAsArrayBuffer(file)
}

export const adjustCanvas = (ctx, width, height, orientation) => {
    // set proper canvas dimensions before transform & export
    if (4 < orientation && orientation < 9) {
      canvas.width = height
      canvas.height = width
    } else {
      canvas.width = width
      canvas.height = height
    }
  // transform context before drawing image
  switch (orientation) {
    case 2:
      ctx.transform(-1, 0, 0, 1, width, 0)
      break
    case 3:
      ctx.transform(-1, 0, 0, -1, width, height)
      break
    case 4:
      ctx.transform(1, 0, 0, -1, 0, height)
      break
    case 5:
      ctx.transform(0, 1, 1, 0, 0, 0)
      break
    case 6:
      ctx.transform(0, 1, -1, 0, height, 0)
      break
    case 7:
      ctx.transform(0, -1, -1, 0, height, width)
      break
    case 8:
      ctx.transform(0, -1, 1, 0, 0, width)
      break
    default:
      break
  }
}